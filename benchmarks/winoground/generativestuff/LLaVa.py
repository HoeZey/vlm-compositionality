import torch 
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import random

class Winoground_generative_evaluation:

    def __init__(self, model_name):
        self.model_name = model_name
        # self.pretrained = pretrained
            
    def show_example(self, benchmark, idx):
        ax1 = plt.subplot(1, 3, 1)
        ax1.title.set_text('image_0')
        plt.imshow(benchmark[idx]["image_0"].convert("RGB"))

        ax2 = plt.subplot(1, 3, 2)
        ax2.title.set_text('image_1')
        plt.imshow(benchmark[idx]["image_1"].convert("RGB"))

        plt.show()

        print("caption_0:", benchmark[idx]["caption_0"])
        print("caption_1:", benchmark[idx]["caption_1"])

    def llava_image_to_caption(self, image, caption_0, caption_1):
        # prompt = "USER: <image>\nDescribe the image in one sentence. ASSISTANT:"

        prompt = "USER: <image>\n Given the image and two candidate captions, which caption is the better description of the given image? (Give a short explanation first, then change to a new line give the final answer in the exact format of: \"The answer is A/B.\")\n"
        prompt += "A. " + caption_0.strip() + "\n"
        prompt += "B. " + caption_1.strip() + "\n"
        prompt += "ASSISTANT:"

        if self.model_name == "llava-hf/llava-1.5-7b-hf":
            model = LlavaForConditionalGeneration.from_pretrained(self.model_name)
            processor = AutoProcessor.from_pretrained(self.model_name)

        inputs = processor(text=prompt, images=image, return_tensors="pt")

        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=15)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
            show_example(idx)

            try:
                ## match caption for image_0
                answer_0 = llava_image_to_caption(image_0, caption_0, caption_1)
                image_to_caption_results[str(idx)+"_image_0"] = answer_0
                print ("\nUsing image_0 to select the better caption: ")
                print (answer_0)
                if "answer is a" in answer_0.lower():
                    correct_a = True
                print ("\n")

                ## match caption for image_1
                answer_1 = llava_image_to_caption(image_1, caption_0, caption_1)
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
        
        return {"LLava_Accuracy": correct / total * 100}
