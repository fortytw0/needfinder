import os
from typing import List
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity 

from sentence_transformers.util import dot_score, cos_sim

from src.experiment2.models.bert import BERT
from src.experiment2.models.longformer import Longformer
from src.experiment2.models.sbert import SBERT
from src.corpus import Corpus



class Experiment(object) : 

    def __init__(
                self,
                max_corpus_size:str,
                corpus_files:List[str],
                query_target_json,
                results_dir,
                ) : 

        self.corpus = self._get_corpus(corpus_files, max_corpus_size)
        self.corpus_size = len(self.corpus)
        self.results_dir = results_dir
        with open(query_target_json , 'r') as f : 
            self.query_targets = json.load(f)


    def _get_model(self, model , **kwargs) : 

        if model=='BERT' : 
            return BERT(**kwargs)        
        elif model=='SBERT' : 
            return SBERT(**kwargs)
        else : 
            return Longformer(**kwargs)

    def _get_sentence_sim(self, query, target, model) :
        sentence_repr = model.encode([query, target])
        return cosine_similarity([sentence_repr[0]] , [sentence_repr[1]])[0 , 0]

    def _get_corpus(self, corpus_files, max_corpus_size) : 
        corpus = Corpus(corpus_files, maxlines=10000)
        corpus.sample(max_corpus_size)
        return list(corpus.data)

    def _get_similarity_function(self, sim) : 

        if sim == 'dot' : 
            return dot_score

        else : 
            return cos_sim


    def _get_relation_df(self , model) : 

        relation_dict = {'query' : [] ,
                'target' : [] , 
                'sim' : [] , 
                }

        for d in self.query_targets : 

            query = d['query']
            target = d['target']
            
            relation_dict['query'].append(query)
            relation_dict['target'].append(target)
            relation_dict['sim'].append(self._get_sentence_sim(query, target, model))
            
        return pd.DataFrame(relation_dict)

    
    def execute(self, model, model_params) : 

        #--- Load Model ---#
        
        print('Initializing model...' , flush=True)
        model = self._get_model(model)
        model.__init__(**model_params)
        print('Finished initializing model...' , flush=True)

        #--- Load Similarity Function ---# 

        print('Loading similarity function...' , flush=True)
        similarity_function = self._get_similarity_function(model_params['sim'])
        print('Loaded similarity function : ' , similarity_function)
        

        #--- Calculate similarity between Query-Targets ---#
        
        print('Calculating similarity between query and targets...' , flush=True)
        relation_df = self._get_relation_df(model)
        print('Finished calculating similarity...' , flush=True)
        
        #--- Encode the corpus ---#
        
        print('Encoding corpus...' , flush=True)
        batch_size = 250
        corpus_repr = []
        for i in tqdm(range(0 , self.corpus_size, batch_size)) : 
            encoding = model.encode(self.corpus[i : i + batch_size])
            print('Encoding Shape : ' + str(encoding.shape) , flush=True)
            corpus_repr.append(encoding)
        corpus_repr = np.concatenate(corpus_repr)
        print('Corpus Repr Shape : ' + str(corpus_repr.shape) , flush=True)
        print('Finished encoding corpus...' , flush=True)
        
        #--- Variables to track our results ---#
        top_5 = {
            'rank_1' : [],
            'rank_2' : [],
            'rank_3' : [], 
            'rank_4' : [],
            'rank_5' : [] 
        }

        target_index = []

        #--- Main Loop ---#        
        for iterindex, row in tqdm(relation_df.iterrows()) : 
            
            #--- Get Target and Query, encode ---#
            
            print('Encoding query/target...' , flush=True)
            target = row['target']
            query = row['query']
            
            print('Target : ' + target , flush=True)
            print('Query : ' + query , flush=True)
            
            target_repr = model.encode([target])
            query_repr = model.encode([query])
            
            #--- Insert target representation at a random index ---#
            
            print('Acquiring random index...' , flush=True)
            index = random.randint(1 , corpus_repr.shape[0]-1)
            
            print('Inserting target into random index in the corpus...' , flush=True)
            inserted = np.concatenate((corpus_repr[0:index, : ] , 
                                    target_repr , 
                                    corpus_repr[index:, : ]))
            
            print('Shape of corpus seeded with target : ' + str(inserted.shape) , flush=True)
            
            #--- Calculate Cosine Similarity, get top_5 preds, get rank of target ---#
            
            print('Calculating similarity between query and inserted_corpus...' , flush=True)
            sim  = similarity_function(inserted , query_repr)  
            print('Similarity : ' ,  sim)
            print('Number of sim : ' , len(sim))

            print('Calculating topk...' , flush=True)
            topk = (-sim).argsort(axis=0)
            
            for iterindex ,i in enumerate(topk[:5]) : 
               
                if i > index : 
                    top_5['rank_{}'.format(iterindex+1)].append(self.corpus[i-1])
                    
                elif i == index : 
                    top_5['rank_{}'.format(iterindex+1)].append(target)
                    
                else :
                    top_5['rank_{}'.format(iterindex+1)].append(self.corpus[i])

            print('Finding the rank of target|query...' , flush=True)
            target_rank = np.where(topk == index)[0]
            target_index.append(target_rank)
                
                
        #--- Saving Results ---#
        
        print('Saving results...' , flush=True)
        for key in top_5.keys() : 
            relation_df[key] = top_5[key]

        relation_df['target_rank'] = target_index

        results_save_path = os.path.join(self.results_dir , model_params['experiment_name'] + '.tsv')
        relation_df.to_csv(results_save_path , index=False , sep='\t')
        
        print('Saved results to ' +  results_save_path , flush=True)
