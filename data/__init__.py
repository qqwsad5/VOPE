from .GenerativeDataset import GenerativeDataset
from .ReCheckDataset import ReCheckDataset

def get_dataset(dataset_name):
    dataset_dict = {
        'generative': GenerativeDataset,
        'recheck': ReCheckDataset,
    }
    return dataset_dict[dataset_name]