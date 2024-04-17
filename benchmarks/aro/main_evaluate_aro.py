import sys
sys.path.append("..")

import pandas as pd
import torch
from torch.utils.data import DataLoader

#-----------------------------------------------------------------------------------
import argparse
import os
import pandas as pd
import torch

from torch.utils.data import DataLoader
import open_clip

# from model_zoo import get_model
# from dataset_zoo import get_dataset
from benchmarks.aro.misc import seed_all, _default_collate, save_scores
from benchmarks.aro.model_zoo.clip_models import CLIPWrapper
from benchmarks.aro.dataset_zoo.aro_datasets import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order
# from benchmark.aro.dataset_zoo.retrieval import COCO_Retrieval, Flickr30k_Retrieval

    
def load_model(model_name, pretrained, device):
    model, _, image_preprocess = open_clip.create_model_and_transforms(model_name, pretrained, device=device)
    model = model.eval()
    clip_model = CLIPWrapper(model, device) 
    return clip_model, image_preprocess

def load_dataset(dataset_name, image_preprocess=None, text_perturb_fn=None, image_perturb_fn=None, download=False, *args, **kwargs):
    """
    Helper function that returns a dataset object with an evaluation function. 
    dataset_name: Name of the dataset.
    image_preprocess: Preprocessing function for images.
    text_perturb_fn: A function that takes in a string and returns a string. This is for perturbation experiments.
    image_perturb_fn: A function that takes in a PIL image and returns a PIL image. This is for perturbation experiments.
    download: Whether to allow downloading images if they are not found.
    """
    if dataset_name == "VG_Relation": 
        from benchmarks.aro.dataset_zoo.aro_datasets import get_visual_genome_relation
        return get_visual_genome_relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "VG_Attribution":
        from benchmarks.aro.dataset_zoo.aro_datasets import get_visual_genome_attribution
        return get_visual_genome_attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "COCO_Order":
        from benchmarks.aro.dataset_zoo.aro_datasets import get_coco_order
        return get_coco_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Flickr30k_Order":
        from benchmarks.aro.dataset_zoo.aro_datasets import get_flickr30k_order
        return get_flickr30k_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    # elif dataset_name == "COCO_Retrieval":
    #     from benchmark.aro.dataset_zoo.retrieval import get_coco_retrieval
    #     return get_coco_retrieval(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    # elif dataset_name == "Flickr30k_Retrieval":
    #     from benchmark.aro.dataset_zoo.retrieval import get_flickr30k_retrieval
    #     return get_flickr30k_retrieval(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def evaluate_open_clip_aro(model_name, pretrained):

    seed = 1
    # model_name = "openai-clip:ViT-B/32"
    # # device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    seed_all(seed)
    # dataset_name = "VG_Relation" 
    dataset_names =["VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order"]
    download = True
    batch_size = 32
    num_workers = 4
    
    model, image_preprocess = load_model(model_name, pretrained, device)
    results = {}

    for dataset in dataset_names:
        dataset = load_dataset(dataset_name, image_preprocess=image_preprocess, download=download)
        
        # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
        collate_fn = _default_collate if image_preprocess is None else None
        
        #batch
        joint_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn) 

        scores = model.get_retrieval_scores_batched(joint_loader)
        result_records = dataset.evaluate_scores(scores)
        df = pd.DataFrame(result_records)

        mean_acc = df['Accuracy'].mean()

        results[dataset] = {
            "Model": model_name,
            "Accuracy": accuracy_mean,
            "Seed": seed
        }   
   
    return results
