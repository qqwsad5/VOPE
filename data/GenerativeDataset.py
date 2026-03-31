import json
import os
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from .BaseDataset import BaseDataset

class GenerativeDataset(BaseDataset):
    def __init__(self, model_for_process, type='captioning', data_root='./Hal_dataset/', **kwargs):
        """
        type: 'captioning','reasoning','writing'
        """
        super().__init__(model_for_process, **kwargs)
        self.name = "generative_{}".format(type)
        self.type = type
        self.data_root = data_root
        with open(os.path.join(self.data_root, './Generative/coco_{}.json'.format(self.type)), 'r') as file:
            self.data = json.load(file)
        self.num_samples = len(self.data)

        # Add data_root prefix to image paths
        for idx in range(self.num_samples):
            self.data[idx]['images'] = os.path.join(self.data_root, 'val2014', self.data[idx]['images'][0])

        self.get_double_word_dict()

    def get_data(self, idx):
        return self.data[idx]

    def get_double_word_dict(self):
        #common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
        
        #Hard code some rules for special cases in MSCOCO
        #qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        #qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']
        
        #double_word_dict will map double words to the word they should be treated as in our analysis
        
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

    def check_synonyms_word(self, word1, word2, similarity_score=0.8):
        token1 = self.nlp(word1)
        token2 = self.nlp(word2)
        similarity = token1.similarity(token2)
        return similarity > similarity_score

    def extract_nouns(self, text):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)

        #match double words
        i = 0
        double_words = []
        idxs = []
        while i < len(tokens):
            idxs.append(i) 
            double_word = ' '.join(tokens[i:i+2])
            if double_word in self.double_word_dict: 
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                i += 1

        tagged = nltk.pos_tag(tokens)
        nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
        return nouns + double_words

    def init_metrics(self):
        metrics = {}
        with open(os.path.join(self.data_root, './Generative/metrics.txt'), "r") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split('=')
            if len(parts) == 2:
                variable_name = parts[0].strip()
                variable_value = eval(parts[1].strip())
                metrics[variable_name] = variable_value

        return metrics

    def calculate_result(self, results, **kwargs):
        """
        results: (json) the output of the dataset stored in json format, with each sample in the format of {"id": XXX, "instruction": XXX, "in_images": XXX, "answer": XXX, "out_image": XXX, ... }
        Return: (dict) all required quantization results in dictionary format
        
        """
        import spacy
        self.nlp = spacy.load("en_core_web_lg")

        metrics = self.init_metrics()
        association = json.load(open(os.path.join(self.data_root, './Generative/relation.json'), 'r', encoding='utf-8'))
        hallucination_words = []
        for word1 in association.keys():
            hallucination_words.append(word1)
            for word2 in association[word1]:
                hallucination_words.append(word2)

        global_safe_words = []

        ground_truth = json.load(open(os.path.join(self.data_root, './Generative/coco_{}.json'.format(self.type)), 'r', encoding='utf-8'))

        for i in tqdm(range(len(results))):

            id = results[i]['id']
            if isinstance(id, str):
                id = int(id)

            # for debug
            record_dict = {}

            nouns = self.extract_nouns(results[i]['answer'])
            # Deduplicate
            nouns = list(dict.fromkeys(nouns))
            after_process_nouns = []
            metrics['avg_out_len'] += len(nouns)
            for noun in nouns:
                if noun in hallucination_words:
                    after_process_nouns.append(noun)
            metrics['avg_filt_rate'] += len(after_process_nouns)/(len(nouns)+1e-8)
            metrics['gen_num'] += 1

            # for debug
            record_dict['output_entity'] = after_process_nouns

            safe_words = []
            safe_list = []
            for idx, word in enumerate(ground_truth[id - 1]['truth']):
                safe_words += association[word]
                safe_list += [idx] * len(association[word])

            safe_words += ground_truth[id - 1]['truth']
            safe_len = len(ground_truth[id - 1]['truth'])
            safe_list += [0] * safe_len
            safe_flag_list = [0] * len(after_process_nouns)

            # for debug
            record_dict['safe_words'] = safe_words
            record_dict['match_safe_words'] = []
            record_dict['not_match_words'] = []

            for idx, noun in enumerate(after_process_nouns):
                if noun in global_safe_words:
                    continue

                if noun in safe_words:
                    # for debug
                    record_dict['match_safe_words'].append(noun)

                    for j in range(len(safe_words)):
                        if noun == safe_words[j]:
                            if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    continue

                flag = False
                for j, check_word in enumerate(safe_words):
                    if self.check_synonyms_word(noun, check_word, similarity_score=0.8):
                        # for debug
                        record_dict['match_safe_words'].append("{}_{}".format(noun, check_word))

                        flag = True
                        if j < (len(safe_list) - safe_len):
                            safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                if flag == True:
                    continue

                # Found an unsafe word (neither in the reference answer nor similar to reference answer words)
                safe_flag_list[idx] = 1
                record_dict['not_match_words'].append(noun)

            metrics['chair_score'] += sum(safe_flag_list)
            metrics['chair_num'] += len(safe_flag_list)
            metrics['safe_cover_score'] += sum(safe_list[-safe_len:])
            metrics['safe_cover_num'] += len(safe_list[-safe_len:])
            if sum(safe_flag_list) == 0:
                metrics['non_hallu_score'] += 1
            metrics['non_hallu_num'] += 1

        res = {}
        CHAIR = round(metrics['chair_score'] / metrics['chair_num'] * 100, 1)
        Cover = round(metrics['safe_cover_score'] / metrics['safe_cover_num'] * 100, 1)
        Ha_p = round(100 - metrics['non_hallu_score'] / metrics['non_hallu_num'] * 100, 1)

        res['chair'] = CHAIR
        res['cover'] = Cover
        res['hal'] = Ha_p

        return res

