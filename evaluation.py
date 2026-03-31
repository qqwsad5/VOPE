import os
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import json

from data import get_dataset
from models import get_model

def evaluation(config):
    # Load model
    model = get_model(config)
    model.model = model.model.to(config.device).eval()

    # Update recheck dataset input model name
    if config.dataset_name == 'recheck':
        config.dataset_info['target_model'] = model.name

    # Load dataset
    dataset = DataLoader(get_dataset(config.dataset_name)(model_for_process=model, data_root=config.dataset_root, **config.dataset_info), batch_size=config.batch_size)

    print("Evaluation: Model {}, Benchmark {}, Batch Size: {}".format(model.name, dataset.dataset.name, config.batch_size))

    # Generate results
    results = []
    with torch.no_grad():
        for raw_data, other_info in tqdm(dataset):
            inputs = model.process(**raw_data)
            inputs = inputs.to(config.device)
            output = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=config.do_sample)
            ans = model.decode(output, skip_special_tokens=True)
            
            for i in range(len(ans)):
                result = {"answer": ans[i]}
                for key in other_info.keys():
                    result[key] = other_info[key][i]
                results.append(result)

    # Save results to file
    save_file_path = os.path.join(config.save_root, dataset.dataset.name)
    os.makedirs(save_file_path, exist_ok=True)
    with open(os.path.join(save_file_path, "{}_results.json".format(model.name)), 'w') as file:
        file.write(json.dumps(results, indent=4))

    # Verify results
    metrics = dataset.dataset.calculate_result(results)
    with open(os.path.join(save_file_path, "{}_metrics.json".format(model.name)), 'w') as file:
        file.write(json.dumps(metrics, indent=4))

    print("Benchmark: {}\nModel: {}".format(dataset.dataset.name, model.name))
    print(metrics)

    # Clear GPU memory
    model.cleanup()

if __name__ == "__main__":
    class DefaultConfigs(object):
        model_root = './models'
        model_name = 'llava'
        dataset_root = './data'
        # generative task
        dataset_name = 'generative'
        dataset_info = {
            'type': 'captioning',
        }
        # # recheck task
        # dataset_name = 'recheck'
        # dataset_info = {
        #     'target_dataset': 'generative_captioning',
        # }

        max_new_tokens = 200
        do_sample = True
        batch_size = 2

        save_root = './results'
        device = 'cuda'

    config = DefaultConfigs()
    evaluation(config)