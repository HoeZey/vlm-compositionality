import torch
import open_flamingo
# from PIL.Image import Image
from torch import FloatTensor
# from typing import Union
from dataclasses import dataclass
from models.model_wrapper_abc import VLMWrapper
from transformers import LlamaTokenizer, AutoModelForCausalLM
from prompts.prompt import Prompt


class OpenFlamingoWrapper(VLMWrapper):
    def __init__(self, device) -> None:
        super().__init__()
        model, preprocess, tokenizer = open_flamingo.create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
        )
        self.device = device
        self.model = model.to(device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left' 

    def _generate_text(self, imgs, captions) -> list[str]:
        img_outputs = []
        for img in imgs:
            for caption in captions:
                img_proc = self.preprocess(img).unsqueeze(0).unsqueeze(1).unsqueeze(0).to(self.device)
                prompt = self.tokenizer([self.prompt.insert_caption(caption)], return_tensors='pt')
                out = self.model.generate(
                    vision_x=img_proc,
                    lang_x=prompt['input_ids'].to(self.device),
                    attention_mask=prompt['attention_mask'].to(self.device),
                    max_new_tokens=self.prompt.max_new_tokens,  
                    num_beams=3,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                img_outputs.append(self.tokenizer.decode(out[0]))
        return img_outputs

    def predict(self, imgs, texts) -> FloatTensor:
        out = self._generate_text(imgs, texts)
        answers = [self.prompt.answered_positive(o) for o in out]
        return torch.tensor(answers).float().view(len(imgs), len(texts)).squeeze()
    
    def set_prompt(self, prompt):
        self.prompt = prompt

    @property
    def name(self) -> str:
        return 'flamingo'

    @property
    def is_contrastive(self) -> str:
        return False


class BLIP2(VLMWrapper):
    def __init__(self):
        self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    def predict(self, images, captions, prompt: Prompt):
        inputs = self.processor(images=images, text=texts, return_tensors='pt').to(device, torch.float16)
        out = self.model(**inputs)
        return 1.0 if prompt.answered_positive(out) else 0.0

    @property
    def is_contrastive(self) -> str:
        return False


class CogVLMWrapper(VLMWrapper):
    def __init__(self, device) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf', low_cpu_mem_usage=True, trust_remote_code=True
        ).to(device).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        # self.tokenizer.padding_side = 'left' 
        self.device = device
        self.torch_type = torch_type

    def _generate_text(self, imgs, texts, prompt) -> tuple[list[int], list[str]]:
        img_outputs = []
        for img in imgs:
            for caption in captions:
                input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=prompt.insert_caption(caption), images=[img])
                inputs = {
                    'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
                    'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
                    'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
                    'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]],
                }
                if 'cross_images' in input_by_model and input_by_model['cross_images']:
                    inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(self.torch_type)]]

                gen_kwargs = {"max_length": 2048, "do_sample": False} # "temperature": 0.9
                
                with torch.no_grad():
                    output = self.model.generate(**inputs, **gen_kwargs)
                    output = output[:, inputs['input_ids'].shape[1]:]
                    output = self.tokenizer.decode(output[0])
                img_outputs.append(self.tokenizer.decode(output[0]))

        return img_outputs

    def predict(self, imgs, texts, prompt) -> float:
        out = self._generate_text(imgs, texts)
        answers = [self.prompt.answered_positive(o) for o in out]
        return torch.tensor(answers).float().view(len(imgs), len(texts)).squeeze()
    
    @property
    def name(self) -> str:
        return 'cogvlm'

    @property
    def is_contrastive(self) -> str:
        return False

    