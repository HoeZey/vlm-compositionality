from tqdm import tqdm
import torch
import open_clip
from datasets import load_dataset


models = [
    ('RN50', 'openai'),
    ('RN101', 'openai'),
    ('RN50x4', 'openai'),
    ('ViT-B-32', 'openai'),
    ('RN50x16', 'openai'),
    ('RN50x64', 'openai'),
    ('ViT-L-14', 'openai'),
    # ('ViT-B-32-quickgelu', 'datacomp_s_s13m_b4k'),
    # ('ViT-B-32-quickgelu', 'datacomp_m_s128m_b4k'),
    # ('ViT-B-16', 'datacomp_l_s1b_b8k'),
    # ('ViT-L-14', 'datacomp_xl_s13b_b90k'),
    ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-g-14', 'laion2b_s12b_b42k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k'),
    ('roberta-ViT-B-32', 'laion2b_s12b_b32k'),
    ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
    ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'),
]


def load_model(model_name, pretrained, device):
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        cache_dir=None,
        device=device
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer, transform


@torch.no_grad()
def text_retrieval(pos_text, neg_text, image, model, tokenizer, transform, device):
    pos_text = tokenizer(pos_text).to(device)
    pos_text_embedding = model.encode_text(pos_text, normalize=True)
    neg_text = tokenizer(neg_text).to(device)
    neg_text_embedding = model.encode_text(neg_text, normalize=True)
    image_embedding = model.encode_image(transform(image).unsqueeze(dim=0).to(device), normalize=True)
    pos_score = pos_text_embedding @ image_embedding.t()
    neg_score = neg_text_embedding @ image_embedding.t()
    return 1 if pos_score.item() > neg_score.item() else 0


def evaluate(dataset, model, tokenizer, transform, device):
    metrics = {}
    for c, data_dict in dataset.items():
        correct_cnt = 0
        for data in tqdm(data_dict, desc=f'evaluating {c}'):
            correct = text_retrieval(data['tested_labels'][0], data['tested_labels'][1], 
                                     data['image'], model, tokenizer, transform, device)
            correct_cnt += correct
        count = len(data_dict)
        metrics[c] = correct_cnt / count
    return metrics


def evaluate_open_clip_on_sugarcrepe(model_name, pretrained):
    """
    return:
    {
        "SugarCrepe_accuracies": {
            "add-obj": x,
            "add_att": y,
            "replace_obj": z,
            "replace_att": a,
            "replace_rel": b,
            "swap_obj": c,
            "swap_att": d
        }
    }
    where
    - add-obj is the accuracy of the model on adding an object to the text
    - add_att is the accuracy of the model on adding an attribute to the text
    - replace_obj is the accuracy of the model on replacing an object in the text
    - replace_att is the accuracy of the model on replacing an attribute in the text
    - replace_rel is the accuracy of the model on replacing a relation in the text
    - swap_obj is the accuracy of the model on swapping objects in the text
    - swap_att is the accuracy of the model on swapping attributes in the text
    """
    auth_token = ""  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token

    sugarcrepe = {
        'add-obj'    : load_dataset("HuggingFaceM4/SugarCrepe_add_obj", use_auth_token=auth_token, trust_remote_code=True)["test"],
        'add_att'    : load_dataset("HuggingFaceM4/SugarCrepe_add_att", use_auth_token=auth_token, trust_remote_code=True)["test"],
        'replace_obj': load_dataset("HuggingFaceM4/SugarCrepe_replace_obj", use_auth_token=auth_token, trust_remote_code=True)["test"],
        'replace_att': load_dataset("HuggingFaceM4/SugarCrepe_replace_att", use_auth_token=auth_token, trust_remote_code=True)["test"],
        'replace_rel': load_dataset("HuggingFaceM4/SugarCrepe_replace_rel", use_auth_token=auth_token, trust_remote_code=True)["test"],
        'swap_obj'   : load_dataset("HuggingFaceM4/SugarCrepe_swap_obj", use_auth_token=auth_token, trust_remote_code=True)["test"],
        'swap_att'   : load_dataset("HuggingFaceM4/SugarCrepe_swap_att", use_auth_token=auth_token, trust_remote_code=True)["test"],
    }

    print(f"Evaluating {model_name}-{pretrained}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, transform = load_model(model_name, pretrained, device)

    acc_val = evaluate(sugarcrepe, model, tokenizer, transform, device)
    return {"SugarCrepe_accuracies": acc_val}