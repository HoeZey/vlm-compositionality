import sys
sys.path.append("..")

import pandas as pd
import torch
from torch.utils.data import DataLoader
# from dataset_zoo import VG_Relation, VG_Attribution , COCO_Order , Flickr30k_Order


#should I include only 2/4 datasets or all 4 inc "COCO_Order", "Flickr30k_Order"?
# def evaluate_open_clip_aro(model_name):
#     # We'll download VG-Relation and VG-Attribution images here. 
#     # Will be a 1GB zip file (a subset of GQA).
#     root_dir="~/.cache" 
#     model, preprocess = get_model(model_name=model_name, device="cuda", root_dir=root_dir)

#     results = {model_name: {}}


#     #Getting the VG-R dataset
#     vgr_dataset = VG_Relation(image_preprocess=preprocess, download=True, root_dir=root_dir)
#     vgr_loader = DataLoader(vgr_dataset, batch_size=16, shuffle=False)

#     # Compute the scores for each test case
#     vgr_scores = model.get_retrieval_scores_batched(vgr_loader)

#     # Evaluate the macro accuracy
#     vgr_records = vgr_dataset.evaluate_scores(vgr_scores)
#     vgr_df = pd.DataFrame(vgr_records)
#     # filtering fr non-symmetric relations
#     symmetric = ['adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing', 'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain', 'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving', 'driving down', 'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing', 'filled with', 'floating in', 'floating on', 'flying', 'flying above', 'flying in', 'flying over', 'flying through', 'full of', 'going down', 'going into', 'going through', 'grazing in', 'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in', 'hanging off', 'hanging over', 'higher than', 'holding onto', 'hugging', 'in between', 'jumping off', 'jumping on', 'jumping over', 'kept in', 'larger than', 'leading', 'leaning over', 'leaving', 'licking', 'longer than', 'looking in', 'looking into', 'looking out', 'looking over', 'looking through', 'lying next to', 'lying on top of', 'making', 'mixed with', 'mounted on', 'moving', 'on the back of', 'on the edge of', 'on the front of', 'on the other side of', 'opening', 'painted on', 'parked at', 'parked beside', 'parked by', 'parked in', 'parked in front of', 'parked near', 'parked next to', 'perched on', 'petting', 'piled on', 'playing', 'playing in', 'playing on', 'playing with', 'pouring', 'reaching for', 'reading', 'reflected on', 'riding on', 'running in', 'running on', 'running through', 'seen through', 'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near', 'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in', 'sleeping on', 'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on', 'standing against', 'standing around', 'standing behind', 'standing beside', 'standing in front of', 'standing near', 'standing next to', 'staring at', 'stuck in', 'surrounding', 'swimming in', 'swinging', 'talking to', 'topped with', 'touching', 'traveling down', 'traveling on', 'tying', 'typing on', 'underneath', 'wading in', 'waiting for', 'walking across', 'walking by', 'walking down', 'walking next to', 'walking through', 'working in', 'working on', 'worn on', 'wrapped around', 'wrapped in', 'by', 'of', 'near', 'next to', 'with', 'beside', 'on the side of', 'around']
#     vgr_df = vgr_df[~vgr_df.Relation.isin(symmetric)]

#     # print(f"VG-Relation Macro Accuracy: {df.Accuracy.mean()}")
#     results[model_name]['VG-Relation'] = vgr_df['Accuracy'].mean()
#     #------------------------------------------------------------

#     # Get the VG-A dataset
#     vga_dataset = VG_Attribution(image_preprocess=preprocess, download=True, root_dir=root_dir)
#     vga_loader = DataLoader(vga_dataset, batch_size=16, shuffle=False)
#     # Compute the scores for each test case
#     vga_scores = model.get_retrieval_scores_batched(vga_loader)

#     # Evaluate the macro accuracy
#     vga_records = vga_dataset.evaluate_scores(vga_scores)
#     vga_df = pd.DataFrame(vga_records)

#     # print(f"VG-Attribution Macro Accuracy: {df.Accuracy.mean()}")
#     results[model_name]['VG-Attribution'] = vga_df['Accuracy'].mean()
#     #---------------------------------------------------------------

#     # Get the COCO-Order dataset
#     coco_dataset = COCO_Order(image_preprocess=preprocess, download=True, root_dir=root_dir)
#     coco_loader = DataLoader(coco_dataset, batch_size=16, shuffle=False)

#     coco_scores = model.get_retrieval_scores_batched(coco_loader)
#     coco_records = coco_dataset.evaluate_scores(coco_scores)

#     coco_df = pd.DataFrame(coco_records)

#     results[model_name]['COCO-Order'] = coco_df['Accuracy'].mean()

#     return results


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

# def config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--device", default="cuda", type=str)
#     parser.add_argument("--batch-size", default=32, type=int)
#     parser.add_argument("--num_workers", default=4, type=int)
#     parser.add_argument("--model_name", default="openai-clip:ViT-B/32", type=str)
#     parser.add_argument("--dataset", default="VG_Relation", type=str, choices=["VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order"])
#     parser.add_argument("--seed", default=1, type=int)
    
#     parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
#     parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
#     parser.add_argument("--output-dir", default="./outputs", type=str)
#     return parser.parse_args()

    
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
    dataset_name = "VG_Relation"
    download = True
    batch_size = 32
    num_workers = 4
    
    model, image_preprocess = load_model(model_name, pretrained, device)
    
    dataset = load_dataset(dataset_name, image_preprocess=image_preprocess, download=download)
    
    # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
    collate_fn = _default_collate if image_preprocess is None else None
    
    #batch
    joint_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn) 

    scores = model.get_retrieval_scores_batched(joint_loader)
    result_records = dataset.evaluate_scores(scores)
    df = pd.DataFrame(result_records)

    for record in result_records:
        record.update({"Model": model_name, "Dataset": dataset_name, "Seed": seed ,'Accuracy': df.Accuracy.mean()})
    
   
   
    # output_file = os.path.join(args.output_dir, f"{args.dataset}.csv")
    # print(f"Saving results to {output_file}")

    # if os.path.exists(output_file):
    #     all_df = pd.read_csv(output_file, index_col=0)
    #     all_df = pd.concat([all_df, df])
    #     all_df.to_csv(output_file)

    # else:
    #     df.to_csv(output_file)
        
    # if args.save_scores:
    #     save_scores(scores, args)

    return result_records

    
# if __name__ == "__main__":
#     args = config()
#     main(args)