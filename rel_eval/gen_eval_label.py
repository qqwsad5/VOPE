import json
import os
from tqdm import tqdm

# List of datasets to evaluate
dataset_list = [
    'generative_captioning'
]

# List of models to evaluate
model_name_list = [
    'llava',
    'qwen3'
]

with open("../data/ReCheck/coco_recheck_question_dict.json", 'r') as file:
    hallu_dict = json.load(file)

reversed_hallu_dict = {}
for key in hallu_dict:
    reversed_hallu_dict[hallu_dict[key]] = key


for dataset in dataset_list:
    for model_name in model_name_list:
        json_model_name = model_name
        with open("../data/ReCheck/{}/{}.json".format(dataset, json_model_name), 'r') as file:
            recheck_info = json.load(file)
        with open("../results/recheck_{}/{}_results.json".format(dataset, json_model_name), 'r') as file:
            recheck_results = json.load(file)
        
        assert len(recheck_info)==len(recheck_results)

        question_list = []
        for i in tqdm(range(len(recheck_info))):
            question_info = recheck_info[i]
            # Only keep entries where the object is not in the image but was mentioned in the answer
            if question_info['label']=='no' and question_info['in_answer']:
                text = recheck_results[i]['answer']
                question_info['ori_answer'] = text
                # Normalize negation expressions and parse the model's yes/no response
                text = text.replace('n\'t', ' not')
                text = text.replace('cannot', 'can not')
                text = text.replace(',', '')
                words = text.split(' ')
                if 'No' in words or 'not' in words or 'no' in words:
                    question_info['answer'] = 'no'
                else:
                    question_info['answer'] = 'yes'
                instruction = question_info['instruction']
                target = reversed_hallu_dict[instruction]
                question_info['target'] = target
                question_list.append(question_info)
            else:
                continue
        os.makedirs('./{}/'.format(dataset), exist_ok=True)
        with open('./{}/{}.json'.format(dataset, json_model_name), 'w', encoding='utf-8') as f:
            json.dump(question_list, f, indent=4)