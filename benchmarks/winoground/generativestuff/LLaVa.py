import torch 
from PIL import Image
import requests
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import random

class Winoground_generative_evaluation:

    def __init__(self, model, processor, prompt_name, evaluation_type):
        self.model = model
        self.processor = processor  
        self.prompt_name = prompt_name  
        self.evaluation_type = evaluation_type  
        # self.pretrained = pretrained
            
    def show_example(self, benchmark, idx):
        ax1 = plt.subplot(1, 3, 1)
        ax1.title.set_text('image_0')
        # plt.imshow(benchmark[idx]["image_0"].convert("RGB"))

        ax2 = plt.subplot(1, 3, 2)
        ax2.title.set_text('image_1')
        # plt.imshow(benchmark[idx]["image_1"].convert("RGB"))

        # plt.show()
        print("index:", idx)
        print("caption_0:", benchmark[idx]["caption_0"])
        print("caption_1:", benchmark[idx]["caption_1"])

    ## define the metrics functions
    def text_correct(self, result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(self, result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(self, result):
        return self.image_correct(result) and self.text_correct(result)


    def llava_image_to_caption(self, image, caption_0, caption_1):
        # prompt = "USER: <image>\nDescribe the image in one sentence. ASSISTANT:"

        if self.prompt_name == "gpt4":
            prompt = "USER: <image>\n Given the image and two candidate captions, which caption is the better description of the given image? (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is A/B.\")\n"
            prompt += "A. " + caption_0.strip() + "\n"
            prompt += "B. " + caption_1.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 15
        
        elif self.prompt_name == "gpt4-moretokens":
            prompt = "USER: <image>\n Given the image and two candidate captions, which caption is the better description of the given image? (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is A/B.\")\n"
            prompt += "A. " + caption_0.strip() + "\n"
            prompt += "B. " + caption_1.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "gpt4-shorterprompt":
            prompt = "USER: <image>\n Given an image and two candidate captions, which caption is the better description of the given image? Give the final answer in the exact format of \"The answer is A/B.\"\n"
            prompt += "A." + caption_0 + "\n"
            prompt += "B." + caption_1 + "\n"  
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        elif self.prompt_name == "choices-first":
            prompt = "USER: <image>\n There are two choices:\n"
            prompt += "A." + caption_0 + "\n"
            prompt += "B." + caption_1 + "\n"  
            prompt += "Given an image and two candidate captions, which caption is the better description of the given image? Give the final answer in the exact format of \"The answer is A/B.\"\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35
        
        elif self.prompt_name == "choices-first-numbers":
            prompt = "USER: <image>\n There are two choices:\n"
            prompt += "1." + caption_0 + "\n"
            prompt += "2." + caption_1 + "\n"  
            prompt += "Given an image and two candidate captions, which caption is the better description of the given image? Give the final answer in the exact format of \"The answer is 1./2..\"\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        else:
            max_new_tokens = 15

        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output = output.split('ASSISTANT:')[1]
        return output


    def llava_image_to_caption_binary_match(self, caption, image):

        if self.prompt_name == "gpt4":
            prompt = "USER: <image>\n Select whether the image matches the caption. Pay close attention to the word order. (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is Yes/No.\"))\n"
            prompt += "Caption: " + caption.strip() + "\n"
            prompt += "ASSISTANT:"
            max_new_tokens = 35

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output = output.split('ASSISTANT:')[1]
        return output    

    def evaluate_winoground_LLava(self):

        auth_token = "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" # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
        winoground = load_dataset("facebook/winoground", use_auth_token=auth_token, trust_remote_code=True)["test"]

        ##Evaluation

        ##images are all winoground images
        random.seed(2023)
        subset_idx = random.sample(range(len(winoground)), 100)
        # len(subset_idx[:20])
        #taking the first 20 for time purposes
        subset_idx = subset_idx[:20]
        if self.evaluation_type == "text_image_group_score":
            text_correct_count = 0
            image_correct_count = 0
            group_correct_count = 0
            total = 0
            image_caption_match_results = {}

            for idx in tqdm(subset_idx):
                image_0 = winoground[idx]["image_0"]
                image_1 = winoground[idx]["image_1"]
                caption_0 = winoground[idx]["caption_0"]
                caption_1 = winoground[idx]["caption_1"]

                print ("Example: #", total)
                self.show_example(benchmark=winoground, idx=idx)
                result = {}
                # try:
                ## map string results to nemurical
                ans_c0_i0 = self.llava_image_to_caption_binary_match(caption_0, image_0)
                image_caption_match_results[str(idx)+"_c0_i0"] = ans_c0_i0
                print ("Match between C0 and I0: ", ans_c0_i0.lower())
                if "answer is yes" in ans_c0_i0.lower():
                    result["c0_i0"] = 1.0
                else:
                    result["c0_i0"] = 0.0

                ans_c0_i1 = self.llava_image_to_caption_binary_match(caption_0, image_1)
                image_caption_match_results[str(idx)+"_c0_i1"] = ans_c0_i1
                print ("Match between C0 and I1: ", ans_c0_i1)
                if "answer is yes" in ans_c0_i1.lower():
                    result["c0_i1"] = 1.0
                else:
                    result["c0_i1"] = 0.0   

                ans_c1_i0 = self.llava_image_to_caption_binary_match(caption_1, image_0)
                image_caption_match_results[str(idx)+"_c1_i0"] = ans_c1_i0
                print ("Match between C1 and I0: ", ans_c1_i0)
                if "answer is yes" in ans_c1_i0.lower():
                    result["c1_i0"] = 1.0
                else:
                    result["c1_i0"] = 0.0

                ans_c1_i1 = self.llava_image_to_caption_binary_match(caption_1, image_1)
                image_caption_match_results[str(idx)+"_c1_i1"] = ans_c1_i1
                print ("Match between C1 and I1: ", ans_c1_i1)
                if "answer is yes" in ans_c1_i1.lower():
                    result["c1_i1"] = 1.0
                else:
                    result["c1_i1"] = 0.0

                print ("result: ", result)

                text_correct_count += 1 if self.text_correct(result) else 0
                image_correct_count += 1 if self.image_correct(result) else 0
                group_correct_count += 1 if self.group_correct(result) else 0

                total += 1
                print ("Current Acc: {}/{} = {}%\n".format(group_correct_count, total, group_correct_count / total * 100))

                # except:
                #     print ("skipped")
                #     continue
            return {"text score": text_correct_count/total*100, "image score": image_correct_count/total*100, "group score": group_correct_count/total*100}
        
        if self.evaluation_type == "accuracy_score":
            correct = 0
            total = 0
            image_to_caption_results = {} ## for saving results

            for idx in tqdm(subset_idx):
                image_0 = winoground[idx]["image_0"]
                image_1 = winoground[idx]["image_1"]
                caption_0 = winoground[idx]["caption_0"]
                caption_1 = winoground[idx]["caption_1"]
                correct_a = False
                correct_b = False

                print ("Example: #", total)
                
                self.show_example(benchmark=winoground, idx=idx)

                try:
                    ## match caption for image_0
                    answer_0 = self.llava_image_to_caption(image_0, caption_0, caption_1)
                    image_to_caption_results[str(idx)+"_image_0"] = answer_0
                    print ("\nUsing image_0 to select the better caption: ")
                    print (answer_0)
                    if "answer is a" in answer_0.lower():
                        correct_a = True
                    print ("\n")

                    ## match caption for image_1
                    answer_1 = self.llava_image_to_caption(image_1, caption_0, caption_1)
                    image_to_caption_results[str(idx)+"_image_1"] = answer_1
                    print ("\nUsing image_1 to select the better caption: ")
                    print (answer_1)
                    if "answer is b" in answer_1.lower():
                        correct_b = True

                    ## the example is counted correct only if both matching are correct
                    if correct_a and correct_b:
                        correct += 1
                    total += 1

                    print ("Current Acc: {}/{} = {}%\n".format(correct, total, correct / total * 100))

                except:
                    print ("skipped")
                    continue
        
            return {"accuracy": correct / total * 100}
        
        else:
            raise ValueError(f"Unknown evaluation type: {self.evaluation_type}")
