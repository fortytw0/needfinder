#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


# Library Imports

import string
import json
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from tqdm import tqdm
import spacy
import pandas as pd
import numpy as np

# Directory Definitions

data_dir = 'data/'

labels_path = 'data/labels.json'
comments_path = 'data/reddits.jsonl'
counter_path = 'data/counter.txt'

# Load pre-trained models

st = SentenceTransformer('paraphrase-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_lg")

# Basic EDA - count number of comments in the reddit dump

comments_file = open(comments_path)
num_comments = 0

for comment in comments_file : 
    num_comments += 1
    
comments_file.close()
comments_file = open(comments_path)
print('Number of comments in file : ' , num_comments)

# Function definitions

def preprocess_string(str1) : 
    
    '''
    For a given string, it strips whitespace, lower-cases, removes punctuation 
    and removes stop words.
    '''
    
    str1 = str1.strip().lower() 
    
    for char in string.punctuation : 
        str1 = str1.replace(char, '')
        
    str1 = str1.split(' ')
    
    str1 = ' '.join([word for word in str1 if word not in nlp.Defaults.stop_words])
        
    return str1

def extract_bigrams(str1) : 
    
    '''
    For a given string, it returns bigrams as a list.
    '''
    
    
    str1_1 = str1[0:]
    str1_2 = str1[1:]
    
    return list(zip(str1_1, str1_2))



def jaq_similarity(str1, str2) : 
    
    '''
    Given 2 strings, it preprocesses them and calculates 
    the Jacquard similarity. 
    '''
    
    str1 = preprocess_string(str1)
    str2 = preprocess_string(str2)
        
    str1 = set(str1.split(' '))
    str2 = set(str2.split(' '))
    
    num_union = len(set.union(str1, str2))
    num_intersection = len(set.intersection(str1, str2))
        
    return num_intersection/num_union


def bigram_jaq_similarity(str1, str2) : 
    
    '''
    For 2 strings, it preprocesses them and calculates 
    the Jacquard similarity between 2 strings. 
    '''
    
    str1 = preprocess_string(str1).split(' ')
    str2 = preprocess_string(str2).split(' ')
        
    str1_bigram = set(extract_bigrams(str1))
    str2_bigram = set(extract_bigrams(str2))
    
    num_union = len(set.union(str1_bigram, str2_bigram))
    num_intersection = len(set.intersection(str1_bigram, str2_bigram))
        
    return num_intersection/num_union


def doc_similarity(str1, str2) : 
    
    '''
    Calculates SpaCy document similarity.
    '''
    
    str1_doc = nlp(preprocess_string(str1))
    str2_doc = nlp(preprocess_string(str2))
    
    return str1_doc.similarity(str2_doc)

def sbert_similarity(str1, str2) : 
    
    '''
    Calculates S-BERT cosine similarity.
    '''
    str1_repr = st.encode(str1)
    str2_repr = st.encode(str2)
    return distance.cosine(qr, comment_repr)
    

# In[ ]:

labels = json.load(open(labels_path))
start_i = int(open(counter_path).read())
print("Starting from index : " , start_i)


# In[ ]:


annotations = []
metrics = {}


for label in labels : 
    metrics[label['title']] = []
    
    
for i in tqdm(range(num_comments)) : 
    
    if i > start_i : 
        line = comments_file.readline()
        comment = json.loads(line)

        num_quotes = len(label['quotes'])


        for label in labels : 
            metrics_dict = {}


            if comment['subreddit'] in label['subreddits'] : 

                for i, quote in enumerate(label['quotes']) : 
                    metrics_dict['sbert_sim_Q{}'.format(i)] = sbert_similarity(quote, comment['body'])
                    
                for i, quote in enumerate(label['quotes']) : 
                    metrics_dict['doc_sim_Q{}'.format(i)] = doc_similarity(quote, comment['body'])
                
                for i, quote in enumerate(label['quotes']) : 
                    metrics_dict['jaq_sim_Q{}'.format(i)] = jaq_similarity(quote, comment['body'])
                
                for i, quote in enumerate(label['quotes']) : 
                    metrics_dict['bigram_jaq_sim_Q{}'.format(i)] = bigram_jaq_similarity(quote, comment['body'])

                metrics_dict['subreddit'] = comment['subreddit']
                metrics_dict['parent_id'] = comment['parent_id']
                metrics_dict['comment'] = comment['body']
                metrics_dict['score'] = comment['score']
                metrics_dict['permalink'] = comment['permalink']
                metrics_dict['created_utc'] = comment['created_utc']


                metrics[label['title']].append(metrics_dict)

        if i%1000 == 0 :
            
            print('Saving : ')

            for label in labels : 
                
                title = label['title']
                print('Working on : ', title)
                
                dest_path = os.path.join(data_dir, title[:10].lower().replace(' ','_').replace('\'', '')+'.json')
                csv_dest_path = dest_path.replace('.json', '_{}-{}.csv'.format(i-1000, i))

                with open(dest_path, 'a') as f : 
                    json.dump(metrics[label['title']], f)    
                print('Finished saving JSON to : ' , dest_path)
                
                pd.DataFrame(metrics[label['title']]).to_csv(csv_dest_path, index=False)
                print('Finished saving CSV to : ' , csv_dest_path)


            with open(counter_path,'w') as f : 
                f.write(str(i))

            start_i = i
                