import json
import pandas as pd
import numpy as np
import os

from src.similarities.arora_similarity import AroraBeam
from src.similarities.cosine_similarity import TFIDFSim
from src.similarities.sbert_similarity import SBERTSim


'''
Script Description : 

This is the main script that will take the Reddit Data and create different similarity comparisons with CHI quotes.

The Reddit data is input as a jsonl file. 

Args : 
- reddit_jsonl_path (str) : path to the Reddit jsonl
- content_field_name (str) : name of the field in the jsonl file that contains the text of the post. 
- id_field_name (str) : name of the field in the jsonl file that should be used as the identifier of the post. Typically the same as the content field. 
- results_dir (str) : where should the resulting csvs be saved?
'''
reddit_jsonl_path = 'data/airbnb_hosts.jsonl'
content_field_name = 'body'
id_field_name = 'body'
results_dir = 'data/results/'

if not os.path.exists(results_dir) :
    os.mkdir(results_dir)


print('Building Arora Beam...')
arora_beam = AroraBeam()
arora_beam._build(reddit_jsonl_path,
                content_field_name, 
                id_field_name)
print('Finished Building Arora Beam...')


print('Building TFIDF...')
tfidf_sim = TFIDFSim()
tfidf_sim.build(reddit_jsonl_path, 
                content_field_name, 
                id_field_name)
print('Finished Building TFIDF...')


print('Building SBERTSim...')
sbert_sim = SBERTSim()
sbert_sim.build(reddit_jsonl_path, 
                content_field_name, 
                id_field_name)
print('Finished Building SBERTSim...')


print('Some assertion checks...')

assert arora_beam.ids == tfidf_sim.ids
assert arora_beam.ids == sbert_sim.ids

with open('data/labels.json') as f : 
    data = json.load(f)
    print('Finished loading data.')
    airbnb_data = data[1]['quotes']
    print('There are : {} quotes from CHI Papers.'.format(len(airbnb_data)))


results = None

for i, abnb in enumerate(airbnb_data) : 

    print(abnb)

    arora_repr = arora_beam.fit([abnb])
    tfidf_repr = tfidf_sim.fit([abnb])
    sbert_repr = sbert_sim.fit([abnb])

    arora_similarity = arora_beam.similarity(arora_repr)
    tfidf_similarity = tfidf_sim.similarity(tfidf_repr)
    sbert_similarity = sbert_sim.similarity(sbert_repr)

    stacked_similarity = np.hstack((arora_similarity, 
                                    tfidf_similarity, 
                                    sbert_similarity))
    


    df = pd.DataFrame(stacked_similarity, index=arora_beam.ids, columns=['Arora' , 'TFIDF' , 'SBERT'])
    df.index.name = abnb

    if len(abnb) > 50 : 
        df.to_csv(os.path.join(results_dir , '{}.csv'.format(i)))

    else : 

        df.to_csv(os.path.join(results_dir , '{}.csv'.format(i)))  


