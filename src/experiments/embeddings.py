import os
import re
import json
import numpy as np
from tqdm import tqdm

from gensim.models import Word2Vec

'''
## User Variables
'''

vocab_fpath = 'data/vocab.txt'
community_jsonl = 'data/airbnb_hosts.jsonl'
output_dir = 'data/'
community_name = 'airbnb_hosts'

'''
## Model Hyper-parameters
'''

num_epochs = int(1e4)
save_path = os.path.join(output_dir, community_name+'.wordvectors')
embedding_dimension = 100
learning_rate = 1e-3


'''
## Function Definitions
'''

def read_vocab(vocab_fpath) -> list :     
    '''
    Reads vocab from pickle file provided in vocab_fpath
    '''

    with open(vocab_fpath) as f : 
        return ['<UNK>'] + f.read().split('\n')

def read_jsonl(jsonl_path , content_field, id_field) : 
    '''
    Reads Reddit jsonl file.
    '''

    content = []
    ids = []

    with open(jsonl_path) as f : 
        reddit_posts = f.readlines()
    
    for post in reddit_posts : 
        post = json.loads(post)
        if content_field in post : 
            content.append(post[content_field])
            ids.append(post[id_field])

    return content , ids


def get_word_index(word, vocab) -> int : 
    
    word = word.strip().lower()

    if word in vocab : 
        return vocab.index(word)

    else : 
        return vocab.index('<UNK>')

def process_sentences(sentences) -> list : 
    processed_sentences = []
    num_sentences = len(sentences)

    for i in tqdm(range(num_sentences)) : 
        sentence = sentences[i].strip().lower()
        tokens = re.findall(r'(\w+)' , sentence)
        tokens = [t for t in tokens if len(t) > 1]
        tokens = [t if t in vocab else '<UNK>' for t in tokens]
        processed_sentences.append(tokens)

    return processed_sentences


'''
## Project Setup

1. Setting up project variables
    a. Read Vocab as list
    b. Read sentences as list
3. Prepare vocab matrix
4. Process sentences into list of words
5. Training
'''

print('Setting up vocab.')
vocab = read_vocab(vocab_fpath)
num_vocab = len(vocab)

print('Setting up processed sentences.')
sentences, ids = read_jsonl(community_jsonl, 'body' , 'id')
processed_sentences = process_sentences(sentences)
num_processed_sentences = len(processed_sentences)

print('Training word2vec model.')
model = Word2Vec(vector_size=embedding_dimension, 
                alpha=learning_rate)

model.build_vocab(processed_sentences, update=False)
model.train(processed_sentences, 
            total_examples=num_processed_sentences, 
            total_words=num_vocab, 
            epochs=num_epochs)


print('Saving.')
model.wv.save(fname=save_path)



if __name__ == '__main__' : 

    print("Analyzing vrbo : ")
    print('Word Embedding : ')
    print(model.wv['vrbo'])

    print('Words most similar to "booking" : ')
    print(model.wv.most_similar('booking' , topn=10))
    
    print('Words most similar to "booking" : ')
    print(model.wv.most_similar('booking' , topn=10))
