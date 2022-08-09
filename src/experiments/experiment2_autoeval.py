import os
import glob
import json
import string

import pandas as pd
import numpy as np
import random

from tqdm import tqdm

'''
1. User defined variables.
'''

eval_dirpath = 'data/eval_set'
eval_results_path = 'data/autoeval_results.csv' 
subreddits = [
            'data/airbnb_hosts.jsonl' 
             ]

results_save_path = 'data/experiment2.csv'

#             'data/airbnb.jsonl' , 
#             'data/vrbo.jsonl'

model_name = 'paraphrase-MiniLM-L3-v2'
model_save_path = 'scratch/summit/dasr8731/needfinder/sentence_transformer_models'


os.environ["SENTENCE_TRANSFORMERS_HOME"] = model_save_path

'''
2. Get sentences from the evaluation set.
'''

groundtruth = []
files = glob.glob(os.path.join(eval_dirpath , '*.json'))
for f in files : 
    groundtruth.append(json.load(open(f)))
    

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


'''
3. Function Definitions
'''

def get_adjacent_quotes(query, section_quotes) : 
    
    query_index = section_quotes.index(query)
    is_last_quote = False
    is_first_quote = False
    
    if query_index == len(section_quotes)-1 : 
        is_last_quote = True
        
    if query_index == 0 : 
        is_first_quote = True
        

    if is_last_quote and is_first_quote : 
        return [None , None]
    
    elif is_last_quote and not is_first_quote:
        return [section_quotes[query_index-1] , None]
    
    elif is_first_quote and not is_last_quote : 
        return [None, section_quotes[query_index+1] ]
    
    else :
        return [section_quotes[query_index-1], section_quotes[query_index+1] ]
    
    
def check_relation(query, target) : 
    
    query_paper_title = quotes2ids[query]['title']
    query_paper_section = quotes2ids[query]['section']
    
    title_quotes = []
    for section_header , quotes in ids2quotes[query_paper_title].items() : 
        title_quotes.extend(quotes)
        
    section_quotes = ids2quotes[query_paper_title][query_paper_section]
    
    adjacent_quotes = get_adjacent_quotes(query, section_quotes)
    
    if target in adjacent_quotes :
        return 'adjacent'
    
    elif target==query : 
        return 'same_quote'
    
    elif target in section_quotes : 
        return 'same_section'
    
    elif target in title_quotes : 
        return 'same_paper' 
    
    else : 
        return 'different_paper'
    

'''
4. Getting relationships DF
'''

results_df  = pd.read_csv(eval_results_path, index_col=0)

relation_dict = {'query' : [] ,
                'target' : [] , 
                'sim' : [] , 
                'relation' : [], 
                }

for query, row in results_df.iterrows() : 
    
    for target , sim in row.items() : 
    
        relation_dict['query'].append(query)
        relation_dict['target'].append(target)
        relation_dict['sim'].append(sim)
        relation_dict['relation'].append(check_relation(query, target))
    
relation_df = pd.DataFrame(relation_dict)


print(relation_df.head())


'''
5. Experiment 2
'''

from src.corpus import Corpus
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


#--- Load model and encode coprus ---#

model = SentenceTransformer(model_name)
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

relation_df.to_csv('data/experiment2_results.csv')
    



