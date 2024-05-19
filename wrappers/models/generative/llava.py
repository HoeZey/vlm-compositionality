from PIL.Image import Image
import torch
from torch import FloatTensor
from transformers import AutoProcessor, LlavaForConditionalGeneration
from wrapper_abc import GenVLMWrapper


class LlaVA(GenVLMWrapper):
    def __init__(self, 
        model_name='llava-hf/llava-1.5-7b-hf', 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ) -> None:
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            device_map='auto',
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def predict_match(self, images: list[Image], captions: list[str]) -> FloatTensor:
        pass

    @torch.no_grad()
    def predict_choice(self, images: list[Image], captions: list[str]) -> FloatTensor:
        return super().predict_choice(images, captions)
    
    # def predict_binary()
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        print(processor.decode(output[0][2:], skip_special_tokens=True))
    
    @property
    def name(self) -> str:
        return 'llava'
