import torch
import open_flamingo
from wrappers.wrapper_abc import VLModelWrapper


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

