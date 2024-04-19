import os
from dotenv import load_dotenv
from datasets import load_dataset


load_dotenv()
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')


def get_winoground_datset():
    return load_dataset('facebook/winoground', token=HF_ACCESS_TOKEN, trust_remote_code=True)


def text_correct(score):
    return score['i0_t0'] > score['i0_t1'] and score['i1_t1'] > score['i1_t0'] 


def image_correct(score):
    return score['i0_t0'] > score['i1_t0'] and score['i1_t1'] > score['i0_t1']


def group_correct(score):
    return image_correct(score) and text_correct(score)
