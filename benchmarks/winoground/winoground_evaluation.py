from tqdm import tqdm
import open_clip
import torch
import numpy as np
from datasets import load_dataset


class Winoground_evaluation:
    """
    This class is used to evaluate the OpenCLIP model on the Winoground dataset
    """
    def __init__(self, model_name, pretrained):
        self.model_name = model_name
        self.pretrained = pretrained

        
    def text_correct(self, result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(self, result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(self, result):
        return self.image_correct(result) and self.text_correct(result)


    def evaluate_open_clip_on_winoground(self):
        """ 
        input: model_name (str) - the name of the model to be used (can be choosen out of open_clip.list_pretrained())
                pretrained (str) - the name of the pre-trined weigths corresponding to the model_name 
        return:
        {
            "Winoground_accuracies": {
                "text_score": x,
                "image_score": y,
                "group_score": z
            }
        }
        where 
        - text_score is the accuracy of the model on matching text with the correct image 
        - image_score is the accuracy of the model on matching image with the correct text
        - group_score is the accuracy of the model if both previous pairings are correct
        """
        auth_token = "" # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
        winoground = load_dataset("facebook/winoground", use_auth_token=auth_token, trust_remote_code=True)["test"]

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model, _, preprocess  = open_clip.create_model_and_transforms(self.model_name, self.pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(self.model_name)

        from tqdm import tqdm
        winoground_clip_scores = []
        for example in tqdm(winoground):
            text0 = example["caption_0"]
            text1 = example["caption_1"]
            text0 = tokenizer(text0).to(device)    
            text1 = tokenizer(text1).to(device)          


            image0 = preprocess(example["image_0"].convert("RGB")).unsqueeze(0).to(device)
            image1 = preprocess(example["image_1"].convert("RGB")).unsqueeze(0).to(device)

            
            with torch.no_grad(), torch.cuda.amp.autocast():
                
                image0_features = model.encode_image(image0)
                image1_features = model.encode_image(image1)
                image0_features /= image0_features.norm(dim=-1, keepdim=True)
                image1_features /= image1_features.norm(dim=-1, keepdim=True)
                # print("doing img norm")

                text0_features = model.encode_text(text0)
                text0_features /= text0_features.norm(dim=-1, keepdim=True)
                text1_features = model.encode_text(text1)
                text1_features /= text1_features.norm(dim=-1, keepdim=True)

                c0_i0 = (100.0 * image0_features @ text0_features.T)
                c0_i1 = (100.0 * image1_features @ text0_features.T)
                c1_i0 = (100.0 * image0_features @ text1_features.T)
                c1_i1 = (100.0 * image1_features @ text1_features.T)


            winoground_clip_scores.append({"id" : example["id"], "c0_i0": c0_i0, "c0_i1": c0_i1, "c1_i0": c1_i0, "c1_i1": c1_i1})

        # evaluating accuracy
    
        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in winoground_clip_scores:
            text_correct_count += 1 if self.text_correct(result) else 0
            image_correct_count += 1 if self.image_correct(result) else 0
            group_correct_count += 1 if self.group_correct(result) else 0

            denominator = len(winoground_clip_scores)
        
        acc_val = {"text_score": text_correct_count/denominator, "image_score": image_correct_count/denominator, "group_score": group_correct_count/denominator}

        return {"Winoground_accuracies": acc_val}



