import torch
import open_clip
from wrappers.wrapper_abc import VLModelWrapper


class OpenClipWrapper(VLModelWrapper):
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
