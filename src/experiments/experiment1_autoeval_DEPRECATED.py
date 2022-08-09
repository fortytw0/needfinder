import os
import glob
import json
import string

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


# Calculating mAP for section level

average_precisions = []
analysis_dict = {'query' : [],
                 'total_targets' : [],
                 
                'candidate_1' : [], 
                'candidate_1_sim' : [],
                'candidate_2' : [] , 
                'candidate_2_sim' : [], 
                'candidate_3' : [] , 
                'candidate_3_sim' : [], 
                'candidate_4' : [] , 
                'candidate_4_sim' : [], 
                'candidate_5' : [] , 
                'candidate_5_sim' : [], 
                 
                'target_1' : [] ,
                'target_1_rank' : [] ,
                'target_1_sim' : [],
                'target_2' : [] ,
                'target_2_rank' : [] ,
                'target_2_sim' : [],
                'target_3' : [] ,
                'target_3_rank' : [] ,
                'target_3_sim' : [],
                'target_4' : [] ,
                'target_4_rank' : [] ,
                'target_4_sim' : [],
                'target_5' : [] ,
                'target_5_rank' : [] ,
                'target_5_sim' : [],
                 
                'average_precision' : []
                }

for query in eval_df.columns : 


    # 1. Extracting groundtruth targets for given query
    target_section = quotes2ids[query]['section']
    target_title = quotes2ids[query]['title']
    target_documents = ids2quotes[target_title][target_section]
    total_targets = len(target_documents)-1
    
    
    
    if total_targets > 0 :
        
        #1a. Adding query and targets to analysis_dict
        analysis_dict['query'].append(query)
        analysis_dict['total_targets'].append(total_targets)


        # 2. Sorting predictions based on similarity
        sorted = eval_df[query].sort_values(ascending=False)
        similarity_scores = sorted.values.tolist()
        predictions = sorted.index.values.tolist()

        # 3. Finding the rank at which a groundtruth target has occured in the prediction
        prediction_indexes = [predictions.index(target) for target in target_documents]
        prediction_indexes.sort()
        prediction_indexes.pop(0)
        
        
        # 3a. Adding the candidates suggested by model to the analysis dictionary
        for i in range(1, 6) : 
            analysis_dict['candidate_{}'.format(i)].append(predictions[i])
            analysis_dict['candidate_{}_sim'.format(i)].append(similarity_scores[i])
            
        
        # 3b. Padding targets to ensure there are exactly 5 targets to look at
        if len(prediction_indexes) > 5: 
            padded_prediction_indexes = prediction_indexes[:5]
        else :
            padded_predition_indexes = prediction_indexes + [None]*(5-len(prediction_indexes))
            
            
        # 3c. Adding 5 targets to analysis dictionary
        for i, pred in enumerate(padded_predition_indexes, start=1) : 
                
            if pred == None : 
                analysis_dict['target_{}'.format(i)].append(None)
                analysis_dict['target_{}_rank'.format(i)].append(None)
                analysis_dict['target_{}_sim'.format(i)].append(None)
                
            else : 
            
                analysis_dict['target_{}'.format(i)].append(predictions[pred])
                analysis_dict['target_{}_rank'.format(i)].append(pred)
                analysis_dict['target_{}_sim'.format(i)].append(similarity_scores[pred])
            
        # 4. Calculating Average Precision for the query
        average_precision = 0
        for k, index in enumerate(prediction_indexes) : 
            average_precision += (k+1)/index
        average_precision /= total_targets
        print(average_precision)
        
        # 4a. Adding average_precision to analysis dictionary
        analysis_dict['average_precision'].append(average_precision)
        

        # 5. Storing the average precision in a large list
        average_precisions.append(average_precision)

    else : 

        print('Encountered no matching quotes...')
        
        
print('Mean Average Precision : ' , np.average(average_precisions))
    

    
    
analysis_df = pd.DataFrame(analysis_dict)
analysis_df.to_csv('data/sectionwise_rank_analysis.csv' , index=False)
    
    

# # calculating mAP for title level 

# average_precisions = []


# for query in eval_df.columns : 


#     # 1. Extracting groundtruth targets for given query
#     target_section = quotes2ids[query]['section']
#     target_title = quotes2ids[query]['title']
#     target_documents = ids2quotes[target_title][target_section]
#     print(query)
#     total_targets = len(target_documents)-1
#     if total_targets > 0 :

#         print(total_targets)

#         # 2. Sorting predictions based on similarity
#         sorted = eval_df[query].sort_values(ascending=False)
#         predictions = sorted.index.values.tolist()

#         # 3. Finding the rank at which a groundtruth target has occured in the prediction
#         prediction_indexes = [predictions.index(target) for target in target_documents]
#         print('Original : ' , prediction_indexes)
#         prediction_indexes.sort()
#         print('After sorting : ' , prediction_indexes)
#         prediction_indexes.pop(0)
#         print('After popping : ' , prediction_indexes)
#         print(predictions[prediction_indexes[0]])
#         print(prediction_indexes)

#         # 4. Calculating Average Precision for the query
#         average_precision = 0
#         for k, index in enumerate(prediction_indexes) : 
#             average_precision += (k+1)/index
#         average_precision /= total_targets
#         print(average_precision)

#         # 5. Storing the average precision in a large list
#         average_precisions.append(average_precision)

#     else : 

#         print('Encountered no matching quotes...')















