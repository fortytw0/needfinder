import os
import glob
import json
import string
from numpy.lib.function_base import average

import pandas as pd
import numpy as np

eval_dirpath = 'data/eval_set'
eval_results_path = 'data/autoeval_results.csv'



# Reading Groundtruth

groundtruth = []
files = glob.glob(os.path.join(eval_dirpath , '*.json'))
for f in files : 
    groundtruth.append(json.load(open(f)))


# Creating maps between quotes - title/section and vice versa

ids2quotes = {}
quotes2ids = {}

for res in groundtruth :
    title = res['title']
    ids2quotes[title] = {}

    for section in res['sections'] : 
        section_header = section['section_header']
        ids2quotes[title][section_header] = section['quotes']

        for quote in section['quotes'] : 

            quotes2ids[quote] = {'title' : title , 
                            'section' : section_header, 
                            }

                    
sentences = list(quotes2ids.keys())

# Encoding sentences

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'paraphrase-MiniLM-L3-v2'
model = SentenceTransformer(model_name)


sentence_repr = model.encode(sentences)

sim = cosine_similarity(sentence_repr, sentence_repr)

eval_df = pd.DataFrame(sim, index=sentences, columns=sentences)
eval_df.to_csv(eval_results_path)


# Calculating mAP

average_precisions = []

for query in eval_df.columns : 


    # 1. Extracting groundtruth targets for given query
    target_section = quotes2ids[query]['section']
    target_title = quotes2ids[query]['title']
    target_documents = ids2quotes[target_title][target_section]
    print(target_documents)
    total_targets = len(target_documents)-1
    print(total_targets)

    # 2. Sorting predictions based on similarity
    sorted = eval_df[query].sort_values(ascending=False)
    predictions = sorted.index.values.tolist()

    # 3. Finding the rank at which a groundtruth target has occured in the prediction
    prediction_indexes = [predictions.index(target)+1 for target in target_documents]
    prediction_indexes.pop(0)
    print(prediction_indexes)

    # 4. Calculating Average Precision for the query
    average_precision = 0
    for k, index in enumerate(prediction_indexes) : 
        average_precision += (k+1)/index
    average_precision /= total_targets
    print(average_precision)

    # 5. Storing the average precision in a large list
    average_precisions.append(average_precision)

    break
    



















