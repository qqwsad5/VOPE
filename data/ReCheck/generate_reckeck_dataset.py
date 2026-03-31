from tqdm import tqdm
import json
import spacy
import os
import nltk
from nltk.stem import WordNetLemmatizer
NLP = spacy.load("en_core_web_lg")

target_dataset = "generative_captioning"
os.makedirs('./{}/'.format(target_dataset), exist_ok=True)
print('Target Dataset: {}'.format(target_dataset))
for target_model in ['llava']:

    with open('../../results/{}/{}_results.json'.format(target_dataset, target_model)) as f:
        results = json.load(f)

    # Determine the mapping between id and image paths
    with open('../Generative/coco_captioning.json', 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    id_dict = {}
    for info in all_data:
        id_dict[int(info['id'])] = info['images']

    #common 'double words' in MSCOCO that should be treated as a single word
    coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']    
    #Hard code some rules for special cases in MSCOCO
    #qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
    animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
    #qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
    vehicle_words = ['jet', 'train']    
    #double_word_dict will map double words to the word they should be treated as in our analysis    
    double_word_dict = {}
    for double_word in coco_double_words:
        double_word_dict[double_word] = double_word
    for animal_word in animal_words:
        double_word_dict['baby %s' %animal_word] = animal_word
        double_word_dict['adult %s' %animal_word] = animal_word
    for vehicle_word in vehicle_words:
        double_word_dict['passenger %s' %vehicle_word] = vehicle_word
    double_word_dict['bow tie'] = 'tie'
    double_word_dict['toilet seat'] = 'toilet'
    double_word_dict['wine glas'] = 'wine glass'

    def extract_nouns(text):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)

        #match double words
        i = 0
        double_words = []
        idxs = []
        while i < len(tokens):
            idxs.append(i) 
            double_word = ' '.join(tokens[i:i+2])
            if double_word in double_word_dict: 
                double_words.append(double_word_dict[double_word])
                i += 2
            else:
                i += 1

        tagged = nltk.pos_tag(tokens)
        nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
        return nouns + double_words

    def check_synonyms_word( word1, word2, similarity_score=0.8):
        token1 = NLP(word1)
        token2 = NLP(word2)
        similarity = token1.similarity(token2)
        return similarity > similarity_score

    # Select question sentence
    with open('./coco_recheck_question_dict.json', 'r') as f:
        hallucination_words_dict = json.load(f)
    def choose_article(noun):
        return hallucination_words_dict[noun]

    association = json.load(open('../Generative/relation.json', 'r', encoding='utf-8'))
    hallucination_words = []
    for word1 in association.keys():
        hallucination_words.append(word1)
        for word2 in association[word1]:
            hallucination_words.append(word2)

    global_safe_words = []

    ground_truth = json.load(open('../Generative/coco_captioning.json', 'r', encoding='utf-8'))

    question_list = []

    print('Generate for {}'.format(target_model))
    for i in tqdm(range(len(results))):

        id = results[i]['id']
        if isinstance(id, str):
            id = int(id)

        nouns = extract_nouns(results[i]['answer'])
        # Deduplicate
        nouns = list(dict.fromkeys(nouns))
        after_process_nouns = []
        for noun in nouns:
            if noun in hallucination_words:
                after_process_nouns.append(noun)

        safe_words = []
        safe_list = []
        for idx, word in enumerate(ground_truth[id - 1]['truth']):
            safe_words += association[word]
            safe_list += [idx] * len(association[word])

        safe_words += ground_truth[id - 1]['truth']
        safe_len = len(ground_truth[id - 1]['truth'])
        safe_list += [0] * safe_len
        safe_flag_list = [0] * len(after_process_nouns)

        for idx, noun in enumerate(after_process_nouns):
            if noun in global_safe_words:
                continue

            yes_flag = False
            if noun in safe_words:
                # Existent
                yes_flag = True
                for j in range(len(safe_words)):
                    if noun == safe_words[j]:
                        if j < (len(safe_list) - safe_len):
                            safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break

            for j, check_word in enumerate(safe_words):
                if check_synonyms_word(noun, check_word, similarity_score=0.8):
                    # Existent
                    yes_flag = True
                    if j < (len(safe_list) - safe_len):
                        safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                    else:
                        safe_list[j] = 1
                    break
            
            # All non-existent cases (anything other than yes is treated as no)
            if yes_flag == False:
                # Non-existent
                question_list.append({"id": id, "images": id_dict[id],  "instruction": choose_article(noun), "label": "no", "in_answer": True})
            else:
                # Existent
                question_list.append({"id": id, "images": id_dict[id],  "instruction": choose_article(noun), "label": "yes", "in_answer": True})

    with open('./{}/{}.json'.format(target_dataset, target_model), 'w', encoding='utf-8') as f:
        json.dump(question_list, f, indent=4)