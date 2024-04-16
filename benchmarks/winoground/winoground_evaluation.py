from tqdm import tqdm
import open_clip
import torch
import numpy as np
from datasets import load_dataset



def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)


def evaluate_open_clip_on_winoground(model_name, pretrained):
    """ 
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
    # auth_token = "" 
    auth_token = ""  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
    winoground = load_dataset("facebook/winoground", use_auth_token=auth_token, trust_remote_code=True)["test"]

    model, _, preprocess  = open_clip.create_model_and_transforms(model_name, pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    from tqdm import tqdm
    winoground_clip_scores = []
    for example in tqdm(winoground):
        text = [example["caption_0"], example["caption_1"]]
        text = tokenizer(text)

        image0 = preprocess(example["image_0"].convert("RGB")).unsqueeze(0)
        image1 = preprocess(example["image_1"].convert("RGB")).unsqueeze(0)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image0)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            output_i0 = (100.0 * image_features @ text_features.T)
            c0_i0 = output_i0[0][0].item()
            c1_i0 = output_i0[0][1].item()

            image_features = model.encode_image(image1)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # output_i1 = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            output_i1 = (100.0 * image_features @ text_features.T)
            c0_i1 = output_i1[0][0].item()
            c1_i1 = output_i1[0][1].item()


        winoground_clip_scores.append({"id" : example["id"], "c0_i0": c0_i0, "c0_i1": c0_i1, "c1_i0": c1_i0, "c1_i1": c1_i1})

    # evaluating accuracy
   
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in winoground_clip_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

        denominator = len(winoground_clip_scores)
    
    acc_val = {"text_score": text_correct_count/denominator, "image_score": image_correct_count/denominator, "group_score": group_correct_count/denominator}

    return {"Winoground_accuracies": acc_val}

