import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .base_model import BaseModel

class LLaVAModel(BaseModel):
    def __init__(self, model_name, model_root, config):
        super().__init__(model_name)
        model_path = os.path.join(model_root, './ckpts/llava-1.5-7b-hf')
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.config = config

        self.NUM_IMG_TOKENS = 576
        self.NUM_PATCHES = 24
        self.IMAGE_TOKEN_INDEX = 32000

    def process(self, img_path: list, text: list, **kwargs):
        '''
        Additional parameters:
        padding='max_length', max_length=1024, truncation=True
        '''
        # Image processing
        img_list = [Image.open(p) for p in img_path]

        # Text processing
        prompt_list = []
        for i in range(len(text)):
            conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": text[i]},
                    {"type": "image"},
                    ],
                },
            ]
            if len(img_list) == 0:
                conversation[0]['content'].pop()
            prompt_list.append(self.processor.apply_chat_template(conversation, add_generation_prompt=True))
        
        inputs = self.processor(images=img_list, text=prompt_list, return_tensors='pt', padding=True, **kwargs).to(torch.float16)

        return inputs

    def generate(self, keep_input=False, **kwargs):
        '''
        Additional parameters:
        max_new_tokens=200, do_sample=True
        '''
        output = self.model.generate(**kwargs)
        if not keep_input:
            input_length = kwargs['input_ids'].shape[1]
            output = output[:, input_length:]
        return output

    def decode(self, output, **kwargs):
        response = self.processor.batch_decode(output, skip_special_tokens=True)
        return response
