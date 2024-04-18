import os
from dotenv import load_dotenv
from datasets import load_dataset


load_dotenv()
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')


def get_winoground_datset():
    return load_dataset('facebook/winoground', token=HF_ACCESS_TOKEN, trust_remote_code=True)


def text_correct(out_img0, out_img1):
    return out_img0[0] > out_img0[1] and out_img1[0] > out_img1[1]


def image_correct(out_img0, out_img1):
    return out_img0[0] > out_img1[0] and out_img0[1] > out_img1[1]


def group_correct(out_img1, out_img2):
    return image_correct(out_img1, out_img2) and text_correct(out_img1, out_img2)
