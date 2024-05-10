import torch
import open_flamingo
# from PIL.Image import Image
# from torch import FloatTensor
# from typing import Union
from dataclasses import dataclass
from models.model_wrapper_abc import VLMWrapper
from transformers import LlamaTokenizer, AutoModelForCausalLM


@dataclass
class Prompt:
    question_prompt: str
    answer_prompt: str
    positive_answer:str

    def insert_caption(self, caption: str) -> str:
        return f'{self.question_prompt}{caption}{self.answer_prompt}'
    
    def answered_positive(self, out: str) -> float:
        return self.positive_answer in out.lower()


class OpenFlamingoWrapper(VLMWrapper):
    def __init__(self, 
            prompt: Prompt,
            vis_encoder='ViT-L-14', 
            vis_pretrained='openai', 
            lang_encoder='anas-awadalla/mpt-1b-redpajama-200b', 
            tokenizer='anas-awadalla/mpt-1b-redpajama-200b',
            xattn_interval=1
        ) -> None:
        super().__init__()
        model, preprocess, tokenizer = open_flamingo.create_model_and_transforms(
            clip_vision_encoder_path=vis_encoder,
            clip_vision_encoder_pretrained=vis_pretrained,
            lang_encoder_path=lang_encoder,
            tokenizer_path=tokenizer,
            cross_attn_every_n_layers=xattn_interval,
        )
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left' 
        self.prompt = prompt

    def _generate_text(self, imgs, texts) -> tuple[list[int], list[str]]:
        imgs_proc = torch.cat([self.preprocess(img).unsqueeze(0) for img in imgs], dim=0).unsqueeze(1).unsqueeze(0)
        texts_proc = self.tokenizer([self.prompt.insert_caption(t) for t in texts], return_tensors='pt')

        print(imgs_proc)
        print(texts_proc)

        out = self.model.generate(
            vision_x=imgs_proc,
            lang_x=texts_proc['input_ids'],
            attention_mask=texts_proc['attention_mask'],
            max_new_tokens=20,
            # num_beams=1,
        )
        return self.tokenizer.decode(out)

    def predict(self, imgs, texts) -> float:
        out = self._generate_text(imgs, texts)
        return 1.0 if self.prompt.answered_positive(out) else 0.0
    
    @property
    def name(self) -> str:
        return 'flamingo'

class CogVLMWrapper(VLMWrapper):
    def __init__(self, 
            device='cpu',
            torch_type='bfloat16'
        ) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf', torch_dtype=torch_type, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(device).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        # self.tokenizer.padding_side = 'left' 
        self.device = device
        self.torch_type = torch_type

    def _generate_text(self, imgs, texts, prompt) -> tuple[list[int], list[str]]:
        texts_proc = [prompt.insert_caption(t) for t in texts]

        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=texts_proc, images=[imgs])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]],
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(self.torch_type)]]

        gen_kwargs = {"max_length": 2048,
                      "do_sample": False} # "temperature": 0.9
        
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
            output = output[:, inputs['input_ids'].shape[1]:]
            output = self.tokenizer.decode(output[0])

        return output.split("</s>")[0]

    def predict(self, imgs, texts, prompt) -> float:
        out = self._generate_text(imgs, texts, prompt)
        print(out)
        return 1.0 if self.prompt.answered_positive(out) else 0.0
    
    @property
    def name(self) -> str:
        return 'cogvlm'

    