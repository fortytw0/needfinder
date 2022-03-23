#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import numpy as np
import glob
import json
import requests
import time


# In[5]:


quotes_path = "data/labels.json"

with open(quotes_path) as f :
    print("Loading labels.json") 
    quotes_dict = json.load(f)
    print("Finished loading labels.json")


# In[6]:


oldest_timestamp_limit = 1332375612 # 22/3/2012


# In[24]:


def get_filename(start_time, end_time, output_dir) : 

	start_time = time.strftime("%Y%m%d%H%M%S",  time.localtime(start_time))
	end_time = time.strftime("%Y%m%d%H%M%S",  time.localtime(end_time))
	return os.path.join(output_dir , "{}_{}.json".format(start_time, end_time))

def save_to_file(start_time, end_time, string_list, output_dir='data') : 

	filename = get_filename(start_time, end_time, output_dir)

	with open(filename, 'w') as f : 
		json.dump(string_list, f)

	return True


# In[14]:


def parse_submission(submission) :
	attributes = ('url', 'upvote_ratio', 'title', 'subreddit', 'selftext', 'score', 'num_comments', 'created', 'created_utc', 'author', 'full_link')
	parsed = {}

	for attribute in attributes :
		if attribute in submission :  
			if type(submission[attribute]) in [float] :  
				parsed[attribute] = str(submission[attribute])
			else : 
				parsed[attribute] = submission[attribute]

		else : 
			parsed[attribute] = None

	# parsed['bert_title'] = get_string_representation(submission['title'])
	parsed_str = json.dumps(parsed)
	return parsed_str


# In[28]:


for quotes in quotes_dict : 
    
    print("Starting to process for paper : " , quotes['title'])
    sleep_time = 0.5
    posts_per_file = 1000
    output_dir = 'data/quotes_matcher'

    url = "https://api.pushshift.io/reddit/search/submission/"
    
    
    for subreddit in quotes['subreddits'] : 
        
        print("Working on subreddit : " , subreddit)
        time_filter = int(time.time())
        oldest_encountered_timestamp = time_filter
        
        save_dir = os.path.join(output_dir, subreddit)
        
        if not os.path.exists(save_dir) :
            os.mkdir(save_dir)
            
            
        while True   : 

            response = requests.get(url, params={
                'subreddit' : subreddit, 
                'before' : time_filter, 
                'limit' : posts_per_file
            })

            parsed_results = []
            num_submissions = 0
            
            print(response.status_code)

            for submission in response.json()['data'] : 

                parsed_results.append(parse_submission(submission))

                if int(submission['created_utc']) < oldest_encountered_timestamp:
                    oldest_encountered_timestamp = int(submission['created_utc'])

                num_submissions += 1

            if num_submissions == 0 : # we have run out of submissions
                break
            elif oldest_encountered_timestamp <= oldest_timestamp_limit: # not interested in posts before a certain year
                break
            else :
                save_to_file(oldest_encountered_timestamp, time_filter, parsed_results, output_dir=save_dir)
                time_filter = oldest_encountered_timestamp

            print('Working on : ' , subreddit)
            print('Saving to : ' , save_dir)
            print('Finished processing from {}'.format(time.strftime("%Y-%m-%d | %H:%M:%S",  time.localtime(time_filter))))
            print("There are {} submissions in this file.".format(num_submissions))

            time.sleep(sleep_time)
