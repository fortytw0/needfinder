from typing import List
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from src.experiment2.main import Experiment


class Search(Experiment) : 

    def __init__(self, max_corpus_size: str, 
                corpus_files: List[str], 
                queries, 
                results_dir):
        
        self.corpus = self._get_corpus(corpus_files, max_corpus_size)
        self.corpus_size = len(self.corpus) 
        self.queries = queries
        self.results_dir = results_dir 

    def execute(self, model, model_params) : 

            #--- Load Model ---#
    
        print('Initializing model...')
        model = self._get_model(model)
        model.__init__(**model_params)
        print('Finished initializing model...')

        #--- Load Similarity Function ---# 

        print('Loading similarity function...' , flush=True)
        similarity_function = self._get_similarity_function(model_params['sim'])
        print('Loaded similarity function : ' , similarity_function)

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

        #--- Variables to Track Result ---#

        top_30 = {'query':[]}
        for i in range(30) : 
            top_30['rank_{}'.format(i + 1)] = []
            top_30['score{}'.format(i + 1)] = []
        


        #--- Search Each Query ---#

        for query in tqdm(self.queries) : 

            query_repr = model.encode([query])

            top_30['query'].append(query)
            sim  = similarity_function(corpus_repr , query_repr)
            topk = (-sim).argsort(axis=0)

            for i , rank in enumerate(topk[:30]) : 

                top_30['rank_{}'.format(i+1)].append(self.corpus[rank])
                top_30['score{}'.format(i+1)].append(sim[rank])

        #--- Save results ---#

        print('Saving results...')
        top_30_df = pd.DataFrame(top_30)


        results_save_path = os.path.join(self.results_dir , model_params['experiment_name']+'_SEARCH' + '.tsv')
        top_30_df.to_csv(results_save_path , index=False , sep='\t')

        print('Saved results to ' +  results_save_path , flush=True)


