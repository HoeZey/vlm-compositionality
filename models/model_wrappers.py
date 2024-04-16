import torch
import open_clip
import open_flamingo 
from open_clip import tokenizer


class OpenClipWrapper(torch.nn.Module):
    def __init__(self, backbone='ViT-B-16', pretrained='openai') -> None:
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        self.model = model
        self.preprocess = preprocess
        
    @torch.no_grad()
    def _encode_inputs(self, img, texts):
        img_feats = self.model.encode_image(self.preprocess(img)).float()
        text_feats = self.model.encode_text(tokenizer.tokenize(texts)).float()
        return img_feats, text_feats
    
    def _classify(img_feats, text_feats):
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        similarity = text_feats @ img_feats.T
        text_probs = (100.0 * similarity).softmax(dim=-1)
        # top_probs, top_labels = txt_probs.topk(5, dim=-1) 
        return text_probs
    
    def forward(self, imgs, texts):
        return self._classify(self._encode_inputs(imgs, texts))
    

class FlamingoWrapper(torch.nn.Module):
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

    def _generate_text(self, imgs, texts):
        imgs = self.preprocess(imgs)
        texts = self.tokenizer(texts)
        out = self.model.generate(
            vision_x=imgs,
            lang_x=texts["input_ids"],
            attention_mask=texts["attention_mask"],
            max_new_tokens=20,
            num_beams=3,
        )
        return out

    def forward(self, imgs, texts):
        out = self._generate_text(imgs, texts)

        