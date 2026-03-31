import json
import os
from .BaseDataset import BaseDataset

class ReCheckDataset(BaseDataset):
    def __init__(self, model_for_process, target_dataset, target_model, data_root='./Hal_dataset/', **kwargs):
        super().__init__(model_for_process)
        self.name = "recheck_{}".format(target_dataset)
        self.data_root = data_root
        self.target_dataset = target_dataset
        self.target_model = target_model

        with open(os.path.join(self.data_root, './ReCheck/{}/{}.json'.format(target_dataset, target_model)), 'r') as file:
            self.data = json.load(file)

        self.num_samples = len(self.data)

        # Add data_root prefix to image paths
        for idx in range(self.num_samples):
            self.data[idx]['images'] = os.path.join(self.data_root, 'val2014', self.data[idx]['images'][0])

    def get_data(self, idx):
        return self.data[idx]

    def calculate_result(self, results, **kwargs):
        """
        results: (json) the output of the dataset stored in json format, with each sample in the format of {"id": XXX, "instruction": XXX, "in_images": XXX, "answer": XXX, "out_image": XXX, ... }
        Return: (dict) all required quantization results in dictionary format
        """
        assert len(results) == len(self.data), "The length of results must be equal to the length of the dataset {}-{}.".format(self.name, self.target_model)
        pred_list = []
        label_list = []
        in_answer_list = []
        for result in results:
            text = result['answer']
            text = text.replace('n\'t', ' not')
            text = text.replace('cannot', 'can not')
            text = text.replace(',', '')
            words = text.split(' ')
            if 'sound' in words:    # Ambiguous answer
                result['answer'] = 'neglect'
                pred_list.append(-1)
            elif 'No' in words or 'not' in words or 'no' in words:
                result['answer'] = 'no'
                pred_list.append(0)
            else:
                result['answer'] = 'yes'
                pred_list.append(1)

        for i in range(len(self.data)):
            if self.data[i]['label'] == 'no':
                label_list.append(0)
            else:
                label_list.append(1)
            # Whether the entity was mentioned in the initial generative answer
            if self.data[i]['in_answer']:
                in_answer_list.append(1)
            else:
                in_answer_list.append(0)

        # Calculate overall metrics (new metrics) ===========================================
        # des: entity present in image; ima: entity absent in image but mentioned by model
        # pos: model predicts present; neg: model predicts absent
        global_level_metrics = {'des_true':0, 'ima_hal':0, 'des_hal':0, 'ima_true':0}
        for i in range(len(self.data)):
            # Only evaluate entities that were mentioned in the model's answer
            if in_answer_list[i] == 0:
                continue
            # Evaluate awareness
            if label_list[i] == 1 and pred_list[i] == 1:
                global_level_metrics['des_true'] += 1
            elif label_list[i] == 1 and pred_list[i] == 0:
                global_level_metrics['ima_hal'] += 1
            elif label_list[i] == 0 and pred_list[i] == 1:
                global_level_metrics['des_hal'] += 1
            elif label_list[i] == 0 and pred_list[i] == 0:
                global_level_metrics['ima_true'] += 1
        des_hallucination = global_level_metrics['des_hal']/(global_level_metrics['des_true']+global_level_metrics['des_hal']+1e-10)
        ima_hallucination = global_level_metrics['ima_hal']/(global_level_metrics['ima_hal']+global_level_metrics['ima_true']+1e-10)
        expression = (global_level_metrics['ima_hal']+global_level_metrics['ima_true']) / \
        (global_level_metrics['des_true']+global_level_metrics['ima_hal']+global_level_metrics['des_hal']+global_level_metrics['ima_true']+1e-10)

        res = {
            'hal_d': des_hallucination,
            'hal_i': ima_hallucination,
            'exp': expression,
        }
        return res