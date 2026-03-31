import torch
from torch.utils.data import Dataset
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, model_for_process, padding=True, sample_idx=None):
        # self.model_for_process = model_for_process
        self.padding = padding
        self.sample_idx = sample_idx

    def __len__(self):
        return len(self.sample_idx) if self.sample_idx is not None else self.num_samples

    def __getitem__(self, idx):
        idx = self.sample_idx[idx] if self.sample_idx else idx
        sample = self.get_data(idx)
        raw_data = {
            'img_path': sample['images'],
            'text': sample['instruction']
        }
        # Other information
        other_info = {}
        for key in sample.keys():
            if key not in ['images', 'instruction']:
                other_info[key] = str(sample[key])

        return raw_data, other_info

    def get_data(self, idx) -> dict:
        return NotImplementedError