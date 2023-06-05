from sentence_transformers import SentenceTransformer, util
import torch
import string
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
        if q != '\n' : 
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

for quote in quotes: 

    quote_embedding = model.encode(quote)
    cos_scores = util.cos_sim(quote_embedding, comment_embeddings)[0]
    top_results = torch.topk(cos_scores, k=30)

    f.write("\n\n======================\n\n")
    f.write("Quote:", quote)
    f.write("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        f.write(comments[idx], "(Score: {:.4f})".format(score))

f.flush()
f.close()

