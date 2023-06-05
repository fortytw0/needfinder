from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import glob

quote_path = 'data/osint_quotes.txt'
comment_path = 'data/osint_comments.txt'

with open(quote_path) as f : 
    quotes = f.read().split('\n') 


with open(comment_path) as f : 
    comments = f.read().split('\n')

def pre_process_quotes(quotes) : 

    print(f'Len of quotes before processing is {len(quotes)}')
    processed_quotes = [] 

    for q in quotes : 
        if q.strip() != '' : 
            processed_quotes.append(q)

    print(f'Len of quotes after processing is {len(processed_quotes)}')

    return processed_quotes


def pre_process_comments(comments) : 
    return comments


quotes = pre_process_quotes(quotes)
comments = pre_process_comments(comments)

model = SentenceTransformer('all-mpnet-base-v2')

# quote_embeddings = model.encode(quotes, convert_to_tensor=True)
comment_embeddings = model.encode(comments, convert_to_tensor=True)

f =  open('data/osint_results.txt' , 'w') 
results = []
result = {}


for quote in quotes: 

    quote_embedding = model.encode(quote)
    cos_scores = util.cos_sim(quote_embedding, comment_embeddings)[0]
    top_results = torch.topk(cos_scores, k=30)

    result['quote'] = quote

    f.write("\n\n======================\n\n")
    f.write("Quote: " + quote)
    f.write("\nTop 5 most similar sentences in corpus:")

    for i , score, idx in enumerate(zip(top_results[0], top_results[1])):
        f.write('\n' + comments[idx] + "(Score: {:.4f})".format(score))
        result[f'rank_{i+1}'] = comments[idx]
        result[f'sim_score_{i+1}'] = score

    results.append(result)
    result = {}

f.flush()
f.close()

pd.DataFrame(results).to_csv('data/osint_results.csv')

