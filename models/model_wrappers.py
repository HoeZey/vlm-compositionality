import torch
import open_clip
import open_flamingo 
from PIL.Image import Image
from open_clip import tokenizer
from torch import FloatTensor
from abc import ABC, abstractmethod
from typing import Union


class VLModelWrapper(ABC, torch.nn.Module):
    '''
    The forward function of VLModelWrapper should return FloatTensor
    containing the logits per text, i.e. out: R^N_text
    '''
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class OpenClipWrapper(VLModelWrapper):
    def __init__(self, backbone='ViT-B-16', pretrained='openai') -> None:
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        self.model = model
        self.preprocess = preprocess
        
    @torch.no_grad()
    def _encode_inputs(self, img: Union[Image, torch.Tensor], texts: list[str]) -> tuple[FloatTensor, FloatTensor]:
        img_feats = self.model.encode_image(self.preprocess(img).unsqueeze(0)).float()
        text_feats = self.model.encode_text(tokenizer.tokenize(texts)).float()
        return img_feats, text_feats
    
    def _classify(self, img_feats, text_feats) -> FloatTensor:
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        similarity = 100 * (text_feats @ img_feats.T)
        return similarity.squeeze(0)
    
    def forward(self, imgs, texts) -> FloatTensor:
        return self._classify(*self._encode_inputs(imgs, texts))
    
    @property
    def name(self) -> str:
        return 'clip'


class OpenFlamingoWrapper(VLModelWrapper):
    def __init__(self) -> None:
        super().__init__()
        model, preprocess, tokenizer = open_flamingo.create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
            cache_dir="PATH/TO/CACHE/DIR"  # Defaults to ~/.cache
        )
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left" 

    def _generate_text(self, imgs, texts) -> tuple[list[int], list[str]]:
        imgs = self.preprocess(imgs)
        texts = self.tokenizer(texts)
        out = self.model.generate(
            vision_x=imgs,
            lang_x=texts["input_ids"],
            attention_mask=texts["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
        )
        return out, self.tokenizer.decode(out)

    def forward(self, imgs, texts) -> list[str]:
        out_raw, out_decoded = self._generate_text(imgs, texts)
        return out_decoded
    
    @property
    def name(self) -> str:
        return 'flamingo'

        