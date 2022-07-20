import os
import glob
import json
import string


eval_dirpath = 'data/eval_set'


def convert_to_id(text) : 

    text = text.lower().strip()
    text = ''.join([char for char in text if char in string.ascii_lowercase])
    text = ''.join([char for char in text if char not in 'aeiou'])
    if len(text) >= 10 : 
        text = text[:10]
    return text


# Getting all quotes and making an entire dictionary

raw_eval_set = []
quotes = {}
ids2quotes = {}

files = glob.glob(os.path.join(eval_dirpath , '*.json'))
print(files)

for f in files : 
    raw_eval_set.append(json.load(open(f)))

for res in raw_eval_set :
    title = res['title']

    for section in res['sections'] : 
        section_header = section['section_header']
        id = convert_to_id(title) + '_' + convert_to_id(section_header)

        for quote in section['quotes'] : 

            quotes[quote] = {'title' : title , 
                            'section' : section_header, 
                            'id' : id}

            if id not in ids2quotes :
                ids2quotes[id] = [quote]
            else : 
                ids2quotes[id].append(quote)


sentences = list(quotes.keys())

print('Num sentences :  ' , len(sentences))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'paraphrase-MiniLM-L3-v2'
model = SentenceTransformer(model_name)


sentence_repr = model.encode(sentences)

print(sentence_repr.shape)


sim = cosine_similarity(sentence_repr, sentence_repr)

print(sim)
print(sim.shape)

import pandas as pd

df = pd.DataFrame(sim, index=sentences, columns=sentences)
df.to_csv('data/autoeval_results.csv')

 
