import json
from PIL import Image



def get_examples():
    with open('./fewshot/captions.json', 'r') as f:
        captions_pairs = json.load(f)

    examples = []
    for captions in captions_pairs.values():
        example = {}
        for i, (img_file, caption) in enumerate(captions.items()):
            example[f'img{i + 1}'] = Image.open(f'./fewshot/images/{img_file}')
            example[f'caption{i + 1}'] = caption
        examples.append(example)

    return examples

print(get_examples())