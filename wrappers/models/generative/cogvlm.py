from PIL.Image import Image
import torch
from torch import FloatTensor
from transformers import LlamaTokenizer, AutoModelForCausalLM
from wrapper_abc import GenVLMWrapper
from wrappers.prompt import Prompt


class CogVLM(GenVLMWrapper):
    def __init__(self, 
        model_name='llava-hf/llava-1.5-7b-hf', 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map='auto',
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=True
        ).eval()
        self.processor = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.torch_dtype = torch_dtype
        self.prompt: Prompt = None

    @torch.no_grad()
    def predict_match(self, images: list[Image], captions: list[str], **gen_kwargs) -> FloatTensor:
        assert self.prompt is not None, 'Please specify a prompt template.'

        answers = []
        for image in images:
            for caption in captions:
                prompt = self.prompt.insert_caption([caption])
                input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=[image])
                inputs = {
                    'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.model.device),
                    'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.model.device),
                    'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.model.device),
                    'images': [[input_by_model['images'][0].to(self.model.device).to(self.torch_type)]],
                }
                # if 'cross_images' in input_by_model and input_by_model['cross_images']:
                #     inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(self.torch_type)]]

                output = self.model.generate(**inputs, **gen_kwargs)
                output = output[:, inputs['input_ids'].shape[1]:]
                output = self.tokenizer.decode(output[0])
                
                answers.append(output.split("</s>")[0])

        return FloatTensor([self.prompt.answered_positive(ans) for ans in answers]).view(2, 2)

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
