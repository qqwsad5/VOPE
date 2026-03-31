import os
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from .base_model import BaseModel

class Qwen3Model(BaseModel):
    def __init__(self, model_name, model_root, config):
        super().__init__(model_name)
        model_path = os.path.join(model_root, './ckpts/Qwen3-VL-8B-Instruct')
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=448*448, padding_side='left')
        self.config = config
    
    def process(self, img_path: list, text: list, **kwargs):
        '''
        Additional parameters:
        padding='max_length', max_length=1024, truncation=True
        '''
        messages = []
        for i in range(len(img_path)):
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path[i],
                        },
                        {"type": "text", "text": text[i]},
                    ],
                }
            ]
            messages.append(message)

        query = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=query,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

        return inputs

    def generate(self, keep_input=False, **kwargs):
        '''
        Additional parameters:
        max_new_tokens=200, do_sample=True
        '''
        if kwargs.get('do_sample', False):
            kwargs['temperature'] = 1.0
            kwargs['top_k'] = 50
            kwargs['top_p'] = 0.9
        output = self.model.generate(**kwargs)
        if not keep_input:
            input_length = kwargs['input_ids'].shape[1]
            output = output[:, input_length:]
        return output

    def decode(self, output, **kwargs):
        response = self.processor.batch_decode(output, skip_special_tokens=True)
        return response