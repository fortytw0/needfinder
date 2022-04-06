#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


# In[ ]:


import json
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from tqdm import tqdm


# In[ ]:


data_dir = 'data/'

labels_path = 'data/labels.json'
comments_path = 'data/reddits.jsonl'
counter_path = 'data/counter.txt'


# In[ ]:


st = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# In[ ]:


comments_file = open(comments_path)
num_comments = 0

for comment in comments_file : 
    num_comments += 1
    
comments_file.close()
comments_file = open(comments_path)
print('Number of comments in file : ' , num_comments)


# In[ ]:


labels = json.load(open(labels_path))


# In[ ]:


start_i = int(open(counter_path).read())


# In[ ]:


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

                quote_repr = st.encode(label['quotes'])

                comment_repr = st.encode(comment['body'])

                for i, qr in enumerate(quote_repr) : 
                    metrics_dict['cosine_Q{}'.format(i)] = distance.cosine(qr, comment_repr)

                metrics_dict['subreddit'] = comment['subreddit']
                metrics_dict['parent_id'] = comment['parent_id']
                metrics_dict['comment'] = comment['body']
                metrics_dict['score'] = comment['score']
                metrics_dict['permalink'] = comment['permalink']
                metrics_dict['created_utc'] = comment['created_utc']


                metrics[label['title']].append(metrics_dict)

        if i%1000 == 0 :

            for label in labels : 
                
                title = label['title']
                dest_path = os.path.join(data_dir, title[:10].lower().replace(' ','_').replace('\'', '')+'.json')

                with open(dest_path, 'w') as f : 
                    json.dump(metrics[label['title']], f)    


            with open(counter_path,'w') as f : 
                f.write(str(i))

            start_i = i
                


# In[ ]:





