from lib2to3.pgen2 import token
import os
import glob
import transformers
import numpy as np

# Setting Program Hyper Parameters 

MODEL_NAME = 'pilot'
model_save_dir = 'models/transformers'

# Tokenizer Information
tokenizer_type = 'wle' # could wle (word level), bpe (byte pair encoding), wpe (word piece)
pretrained_tokenizer_model = None # leave none if you do not want to use a pre-trained tokenizer

log_path = 'logs/train.log'

corpus_files = ['data/airbnb_hosts.jsonl' , 'data/airbnb.jsonl' , 'data/vrbo.jsonl']
content_field = 'body'
maxlines = 1e5
train_split_ratio = 0.8 # Ratio of samples to consider in train 

# Setup logging

import logging

log_format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename = log_path,
                    filemode = "w",
                    format = log_format, 
                    level = logging.INFO)

logger = logging.getLogger()

logger.debug('Logging has been setup.')


# Checking if hyper parameters are valid 

from src.utils import create_dir_if_not_exist

create_dir_if_not_exist(model_save_dir)

tokenizer_save_path = os.path.join(model_save_dir , MODEL_NAME + '.json')

# Read Dataset , Make splits

from src.utils import read_jsonl

logger.info('Reading dataset...')

data = []
for corpus_file in corpus_files : 
        data.extend(read_jsonl(corpus_file, field=content_field, max_lines=maxlines))

num_data = len(data)

logger.info('Finished reading dataset...')
logger.info('There are {} samples to train on...'.format(num_data))

num_train_data = int(train_split_ratio*num_data)

train_data = data[:num_train_data]
val_data = data[num_train_data:]

num_val_data = len(val_data)

logger.info('Number of train samples :  {}.'.format(num_train_data))
logger.info('Number of eval samples : {}.'.format(num_val_data))

# Setting up tokenizer

from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer, BpeTrainer
from tokenizers.models import WordLevel, WordPiece, BPE
from tokenizers.pre_tokenizers import Digits, Whitespace, Punctuation, Sequence
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import BertNormalizer

if pretrained_tokenizer_model is None : 

    logger.info('Training tokenizer from scratch...')

    if tokenizer_type == 'wle' : 
        trainer = WordLevelTrainer
        model = WordLevel
    elif tokenizer_type == 'bpe' :
        trainer = BpeTrainer
        model = BPE
    elif tokenizer_type == 'wpe' : 
        trainer = WordPieceTrainer
        model = WordPiece

    else : 
        logging.error('There is an error with your tokenizer selection.')

    tokenizer = Tokenizer() 
    tokenizer.normalizer = BertNormalizer()
    tokenizer.pre_tokenizer = Sequence(Whitespace(), Digits(individual_digits=True), Punctuation(behavior='removed'))
    tokenizer.post_processor = BertProcessing(sep=('<\s>' , 1) , cls=('<s>' , 0))
    tokenizer.train_from_iterator(train_data , trainer=trainer(special_tokens=['<s>' , '<\s>' , '<unk>' , '<pad>' , '<mask>']))
    tokenizer.save(tokenizer_save_path)

    logger.info('Finished training tokenizer from scratch...')

else : 
    logger.info('Loading Pre Trained Tokenizer Model')
    tokenizer = Tokenizer().from_pretrained(pretrained_tokenizer_model)
    logger.info('Finished loading Pre Trained Tokenizer Model')


logger.info('Testing tokenizer on sample sequence "Life, The Universe and Everything"')
logger.info(tokenizer.encode('Life, The Universe and Everything'))