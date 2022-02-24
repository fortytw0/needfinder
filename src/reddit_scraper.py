import tensorflow as tf
import numpy as np
import os
import glob
import json
import requests
import time

import logging

logging.basicConfig(handlers=[
					logging.FileHandler('logs/scraper.log'),
					logging.StreamHandler()], 
					level=logging.DEBUG, 
					format='[%(asctime)s]  [%(levelname)s] : %(message)s')



'''
Section 1 : BERT Initialization with Hugging Face
'''

from transformers import BertConfig, BertTokenizer, TFBertModel

try : 
	bert = TFBertModel.from_pretrained('/projects/dasr8731/needfinder/models/bert-base-uncased', local_files_only=True)
	tokenizer = BertTokenizer.from_pretrained('/projects/dasr8731/needfinder/models/bert-base-uncased', local_files_only=True)
	# bert = TFBertModel.from_pretrained('bert-base-uncased')
	# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	sample_text = 'When the levee breaks, you got not place to stay.'
	tokens = tokenizer(sample_text, return_tensors="tf")
	model_output = bert(tokens)
	cls = model_output.last_hidden_state[0, 0, :]

	assert cls.numpy().shape == (768, )
  
	logging.debug("BERT Initialized successfully...")

except Exception as e : 

	logging.error("There was an error in initializing BERT : {}".format(e))


def get_string_representation(string) : 
	tokens = tokenizer(string, return_tensors="tf")
	model_output = bert(tokens)
	cls = model_output.last_hidden_state[0, 0, :]
	return str(list(cls.numpy()))

'''
Section 2 : PushShift API calls for getting data
'''


sleep_time = 1
subreddit = 'fantasyfootball'
time_filter = int(time.time())
oldest_created_utc = time_filter
more_posts_exist = True
posts_per_file = 1000
output_dir = '/scratch/summit/dasr8731/needfinder'

url = "https://api.pushshift.io/reddit/search/submission/"


if not os.path.exists(output_dir) :
	logging.debug("{} : This directory did not exist before. Creating a new directory.".format(output_dir))
	os.mkdir(output_dir)

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

	parsed['bert_title'] = get_string_representation(submission['title'])
	parsed_str = json.dumps(parsed)
	return parsed_str

def get_filename(start_time, end_time) : 

	start_time = time.strftime("%Y%m%d%H%M%S",  time.localtime(start_time))
	end_time = time.strftime("%Y%m%d%H%M%S",  time.localtime(end_time))
	return os.path.join(output_dir , "{}_{}.jsonl".format(start_time, end_time))

def save_to_file(start_time, end_time, string_list) : 

	filename = get_filename(start_time, end_time)

	with open(filename, 'w') as f : 
		f.writelines(string_list)

	return True

i = 0
while more_posts_exist : 

	response = requests.get(url, params={
						  'subreddit':subreddit,
                          'before':time_filter, 
                          'posts_per_file':100})

	parsed_results = []

	num_submissions = 0
	for submission in response.json()['data'] : 

		parsed_results.append(parse_submission(submission))

		if int(submission['created_utc']) < oldest_created_utc:
			oldest_created_utc = int(submission['created_utc'])

		num_submissions += 1

	if num_submissions == 0 :
		more_posts_exist = False

	save_to_file(oldest_created_utc, time_filter, parsed_results)
	time_filter = oldest_created_utc

	logging.debug("Finished processing upto : {}".format(time.strftime("%Y-%m-%d | %H:%M:%S",  time.localtime(time_filter))))
	time.sleep(sleep_time)

	i += 1

	if i>2 : 
		break

