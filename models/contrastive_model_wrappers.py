import torch
import open_clip
from PIL.Image import Image
from torch import FloatTensor
from typing import Union
from model_wrapper_abc import VLMWrapper


class OpenClipWrapper(VLMWrapper):
    def __init__(self, backbone='ViT-B-32', pretrained='openai') -> None:
        super().__init__()
        
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(backbone)
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = device
        
    @torch.no_grad()
    def _encode_inputs(self, img: Union[Image, torch.Tensor], texts: list[str]) -> tuple[FloatTensor, FloatTensor]:
        img_feats = self.model.encode_image(self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device))
        # text_feats = self.model.encode_text(tokenizer.tokenize(texts)).float()
        text_feats = self.model.encode_text(self.tokenizer(texts).to(self.device))

        return img_feats, text_feats
    
    def _classify(self, img_feats, text_feats) -> FloatTensor:
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        similarity = (100 * img_feats @ text_feats.T)
        return similarity.squeeze(0)

    def predict(self, imgs: list[Image], texts: list[str]) -> FloatTensor:
        return self._classify(*self._encode_inputs(imgs, texts))
    
    @property
    def name(self) -> str:
        return 'clip'


class OpenFlamingoWrapper(VLMWrapper):
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

        