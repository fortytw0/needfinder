from operator import index
import pandas as pd
import numpy as np

import os
import glob
import json
import string

eval_results_path = 'data/autoeval_results.csv'

eval_df = pd.read_csv(eval_results_path, index_col=0)

eval_dirpath = 'data/eval_set'


def convert_to_id(text) : 

    text = text.lower().strip()
    text = ''.join([char for char in text if char in string.ascii_lowercase])
    text = ''.join([char for char in text if char not in 'aeiou'])
    if len(text) >= 10 : 
        text = text[:10]
    return text


# Getting all quotes and making an entire dictionary

raw_eval_set = []
quotes = {}
ids2quotes = {}

files = glob.glob(os.path.join(eval_dirpath , '*.json'))


for f in files : 
    raw_eval_set.append(json.load(open(f)))

for res in raw_eval_set :
    title = res['title']

    for section in res['sections'] : 
        section_header = section['section_header']
        id = convert_to_id(title) + '_' + convert_to_id(section_header)

        for quote in section['quotes'] : 

            quotes[quote] = {'title' : title , 
                            'section' : section_header, 
                            'id' : id}

            if id not in ids2quotes :
                ids2quotes[id] = [quote]
            else : 
                ids2quotes[id].append(quote)


sentences = list(quotes.keys())

sentence = sentences[0]

i = 0

predictions = []

for column in eval_df.columns : 

    query = {'quote' : column}
    query['id'] = quotes[column]['id']
    query['title'] = quotes[column]['title']
    query['section'] = quotes[column]['section']

    total_id_matches = 0
    total_title_matches = 0
    total_section_matches = 0
    predicted_targets = []

    sorted = eval_df[column].sort_values(ascending=False)

    for i , s in sorted[:6].iteritems() : 

        predicted_target = {'quote' : i , 'similarity_score' : str(round(s , 3))}
        predicted_target['id'] = quotes[i]['id']
        predicted_target['title'] = quotes[i]['title']
        predicted_target['section'] = quotes[i]['section']

        if predicted_target['title'] == query['title'] : 
            predicted_target['title_matched'] = True
            total_title_matches += 1
        else : 
            predicted_target['title_matched'] = False

        if predicted_target['section'] == query['section'] : 
            predicted_target['section_matched'] = True
            total_section_matches += 1
        else : 
            predicted_target['section_matched'] = False

        predicted_targets.append(predicted_target)

    

    prediction = {'query' : query , 
                'total_title_matches' : total_title_matches , 
                'total_section_matches' : total_section_matches,
                'predicted_targets' : predicted_targets
                }

    predictions.append(prediction)
        


with open('data/processed_results.json', 'w') as f: 
    json.dump(predictions, f)

    

    