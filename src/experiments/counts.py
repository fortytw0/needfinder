import os
os.chdir('/scratch/summit/dasr8731/needfinder')

import pandas as pd
import numpy as np
import json
import glob

'''
Datasets to consider. 
'''

datapaths = {
    'airbnb' : 'data/airbnb_hosts.jsonl',
    'truegaming' : 'data/truegaming.jsonl',
    'gaming' : 'data/gaming.jsonl',
    'caloriecount' : 'data/caloriecount.jsonl',
}

num_datasets = len(datapaths)

'''
Function definitions. 
'''

def read_jsonl_data(data_path) : 
    
    data = []
    
    with open(data_path) as f : 
        for line in f.readlines() : 
            sample = json.loads(line)
            data.append(sample['body'])

    print('...Identified {} posts in {}.'.format(len(data) , data_path))
    
    return data


'''
Reading data from disk. 
'''

datasets = {}


for subreddit, data_path in datapaths.items() :
    datasets[subreddit] = read_jsonl_data(data_path)
    
print('Finished reading files from disk.\n')    

'''
Creating Unigram counts
'''


from sklearn.feature_extraction.text import CountVectorizer

unigram_counts = {}
subreddit_vectorizers = {}

for subreddit, dataset in datasets.items() : 
    
    vectorizer = CountVectorizer(max_features=int(1e4))
    X = vectorizer.fit_transform(dataset).toarray().sum(axis=0)
    subreddit_vectorizers[subreddit] = vectorizer
    unigram_counts[subreddit] = {word:count for word,count in zip(vectorizer.get_feature_names(), X)}
    print('...Fit data for subreddit : ' , subreddit)
    
print('Finished creating vectorizer for all subreddits.\n')


'''
Identifying Overall Vocabulary
'''

vocab = []

for subreddit, vectorizer in subreddit_vectorizers.items() : 
    
    subreddit_vocab = vectorizer.get_feature_names()
    print('...Identified {} words in {}.'.format(len(subreddit_vocab) , subreddit))
    vocab += subreddit_vocab


vocab = list(set(vocab))
num_words = len(vocab)
print('Identified {} unique words across all subreddits.'.format(num_words))


'''
Building Numpy Array
'''

term_frequency_matrix = np.zeros((num_words, num_datasets))

i = 0 




for subreddit, word_counts in unigram_counts.items() :
    
    for word, count in word_counts.items() : 
            
        j = vocab.index(word)
        term_frequency_matrix[j , i] = count
        
    i += 1
        
        
term_frequency_matrix += 1

'''
Saving Term Frequency Matrix
'''

df = pd.DataFrame(term_frequency_matrix, index=vocab, columns=list(unigram_counts.keys()))

df.to_csv('data/term_frequency_matrix.csv')

term_probability_matrix = term_frequency_matrix / np.sum(term_frequency_matrix, axis=0)

df = pd.DataFrame(term_probability_matrix, index=vocab, columns=list(unigram_counts.keys()))

df.to_csv('data/term_probability_matrix.csv')


from scipy.special import softmax

print('term_probability_matrix.shape : ' , term_probability_matrix.shape)

softmax_matrix = softmax(term_probability_matrix, axis=0)

df = pd.DataFrame(softmax_matrix, index=vocab, columns=list(unigram_counts.keys()))

df.to_csv('data/softmax_matrix.csv')

    
    
    
    
