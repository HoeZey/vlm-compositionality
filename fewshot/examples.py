import json
from PIL import Image


def get_examples():
    with open('./fewshot/captions_dalle.json', 'r') as f:
        captions_pairs = json.load(f)

    # examples = []
    # for captions in captions_pairs.values():
    #     example = {}
    #     for i, (img_file, caption) in enumerate(captions.items()):
    #         example[f'img{i + 1}'] = Image.open(f'./fewshot/images/{img_file}')
    #         example[f'caption{i + 1}'] = caption
    #     examples.append(example)


    synthetic_examples = []
    for captions in captions_pairs.values():
        example = {}
        for i, (img_file, capts) in enumerate(captions.items()):
            example['image'] = Image.open(f'./fewshot/images/{img_file}')
            example['caption_A'] = capts[0]
            example['caption_B'] = capts[1]
        synthetic_examples.append(example)

    # self.synthetic_examples = synthetic_examples
    return synthetic_examples

