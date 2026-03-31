import os
import torch
from .llava_model import LLaVAModel
from .qwen3_model import Qwen3Model

def get_model(config):
    models = {
        'llava': LLaVAModel,
        'qwen3': Qwen3Model,
    }
    return models[config.model_name](config.model_name, config.model_root, config)
