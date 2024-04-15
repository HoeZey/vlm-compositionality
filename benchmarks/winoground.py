import os
import matplotlib.pyplot as plt
from random import randint
from dotenv import load_dotenv
from datasets import load_dataset


load_dotenv()
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')


def get_winoground_datset():
    return load_dataset('facebook/winoground', token=HF_ACCESS_TOKEN, trust_remote_code=True)


if __name__ == '__main__':
    winoground = get_winoground_datset()['test']
    idx = randint(0, winoground.num_rows - 1)
    example = winoground[idx]

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(example['image_0'])
    axs[1].imshow(example['image_1'])
    axs[0].set_title(example['caption_0'])
    axs[1].set_title(example['caption_1'])
    plt.tight_layout()
    plt.show()
