import os
import glob
import json
import string
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
import random

from tqdm import tqdm

'''
1. User defined variables.
'''

eval_dirpath = 'data/eval_set'
query_target_json = 'annotations/query_targets.json' 
subreddits = [
            'data/airbnb_hosts.jsonl' 
             ]



model_name = 'paraphrase-MiniLM-L3-v2'
model_save_path = './sentence_transformer_models'

results_save_path = 'data/experiment2/{}.csv'.format(model_name)

# Loading model

from sentence_transformers import SentenceTransformer


os.environ["SENTENCE_TRANSFORMERS_HOME"] = model_save_path
model = SentenceTransformer(model_name)

'''
2. Get sentences from the evaluation set.
'''
with open(query_target_json , 'r') as f : 
    query_targets = json.load(f)

'''
3. Function Definitions
'''

def get_sentence_sim(query, target) :

    sentence_repr = model.encode([query, target])
    return cosine_similarity([sentence_repr[0]] , [sentence_repr[1]])[0 , 0]
    

'''
4. Getting relationships DF
'''

relation_dict = {'query' : [] ,
                'target' : [] , 
                'sim' : [] , 
                }

for d in query_targets : 

    query = d['query']
    target = d['target']
    
    relation_dict['query'].append(query)
    relation_dict['target'].append(target)
    relation_dict['sim'].append(get_sentence_sim(query, target))
    
relation_df = pd.DataFrame(relation_dict)

print(relation_df.head())



'''
5. Experiment 2
'''

from src.corpus import Corpus



#--- Load model and encode coprus ---#


corpus = Corpus(subreddits)
corpus_repr = model.encode(corpus.data)

print('Finished encoding model...')
print(corpus_repr.shape)

#--- Finding the top 5 retrieved items, and the rank of the index ---#


top_5 = {
    'rank_1' : [],
    'rank_2' : [],
    'rank_3' : [], 
    'rank_4' : [],
    'rank_5' : [] 
}


target_index = []

#--- Experiment 2 Main Loop ---#


for iterindex, row in tqdm(relation_df.iterrows()) : 
    
    #--- Get Target and Query, encode ---#
    target = row['target']
    query = row['query']
    
    target_repr = model.encode([target])
    query_repr = model.encode([query])
    
    #--- Insert target representation at a random index ---#
    index = random.randint(1 , corpus_repr.shape[0]-1)
    
    inserted = np.concatenate((corpus_repr[0:index, : ] , 
                               target_repr , 
                               corpus_repr[index:, : ]))
    
    
    #--- Calculate Cosine Similarity, get top_5 preds, get rank of target ---#
    sim  = cosine_similarity(inserted, query_repr)  
    topk = (-sim).argsort(axis=0)
    
    for iterindex ,i in enumerate(topk[:5]) : 
        
        i = i[0]
        
        if i > index : 
            top_5['rank_{}'.format(iterindex+1)].append(corpus.data[i-1])
            
        elif i == index : 
            top_5['rank_{}'.format(iterindex+1)].append(target)
            
        else :
            top_5['rank_{}'.format(iterindex+1)].append(corpus.data[i])
            
    target_rank = np.where(topk == index)[0]
    target_index.append(target_rank)
        
        
        
for key in top_5.keys() : 
    relation_df[key] = top_5[key]
    
relation_df['target_rank'] = target_index
    
print('relation_df after experiment 2 : ')
print(relation_df.head())

relation_df.to_csv(results_save_path)
    



