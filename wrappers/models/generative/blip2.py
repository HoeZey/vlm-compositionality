from PIL.Image import Image
import torch
from torch import FloatTensor
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from wrapper_abc import GenVLMWrapper


class BLIP2(GenVLMWrapper):
    def __init__(self, 
        model_name='Salesforce/blip2-opt-2.7b', 
        torch_dtype=torch.float16,
        load_in_8bit=True
    ) -> None:
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            device_map='auto',
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=load_in_8bit
        ).eval()
        self.processor = Blip2Processor.from_pretrained(model_name)

    @torch.no_grad()
    def predict_match(self, images: list[Image], captions: list[str]) -> FloatTensor:
        pass

    @torch.no_grad()
    def predict_choice(self, images: list[Image], captions: list[str]) -> FloatTensor:
        return 
    
    # def predict_binary()
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        print(processor.decode(output[0][2:], skip_special_tokens=True))
    
    @property
    def name(self) -> str:
        return 'llava'
