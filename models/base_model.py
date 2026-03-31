import torch

class BaseModel():
    def __init__(self, model_name):
        self.name = model_name
    
    def process(self, img_path: str, text: str, **kwargs):
        return NotImplementedError

    def generate(self, **kwargs):
        return NotImplementedError

    def decode(self, output, **kwargs):
        return NotImplementedError
    
    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()