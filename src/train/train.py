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
pretrained_tokenizer_model = 'distilroberta-base' # leave None if you do not want to use a pre-trained tokenizer

# Transformer Information
pretrained_transformer_model = 'distilroberta-base' # the transfomer architecture to use for training


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

    tokenizer = Tokenizer(model()) 
    tokenizer.normalizer = BertNormalizer()
    tokenizer.pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True), Punctuation(behavior='removed')])
    tokenizer.post_processor = BertProcessing(sep=('<\s>' , 1) , cls=('<s>' , 0))
    tokenizer.train_from_iterator(train_data , 
                                  trainer=trainer(special_tokens=['<s>' , '<\s>' , '<unk>' , '<pad>' , '<mask>']),
                                  length=num_train_data)
    
    tokenizer.save(tokenizer_save_path)

    logger.info('Finished training tokenizer from scratch...')

else : 
    from transformers import AutoTokenizer
    logger.info('Loading Pre Trained Tokenizer Model')
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_model)
    logger.info('Finished loading Pre Trained Tokenizer Model')


logger.info('Testing tokenizer on sample sequence "Life, The Universe and Everything"')
logger.info(tokenizer('Life, The Universe and Everything'))

# Setting up dataset

from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling


train_dataset = Dataset.from_dict({'text' : train_data})
val_dataset = Dataset.from_dict({'text' : val_data})


if pretrained_tokenizer_model is None : 

    def tokenizer_function(example) : 
        tokenized = {}

        encoded = tokenizer.encode(example['text'])
        tokenized['ids'] = encoded.ids
        tokenized['type_ids'] = encoded.type_ids
        tokenized['tokens'] = encoded.tokens
        tokenized['attention_mask'] = encoded.attention_mask
        tokenized['token_types'] = encoded.type_ids

        return tokenized

    tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=False)
    tokenized_val_dataset = val_dataset.map(tokenizer_function, batched=False)
    
else : 
    
    def tokenizer_function(example) : 
        encoded = tokenizer(example['text'], return_tensors='tf', padding='max_length', truncation=True)
        encoded['input_ids'] = encoded.input_ids[0]
        encoded['attention_mask'] = encoded.attention_mask[0]
        encoded['labels'] = encoded.input_ids
        return encoded

    tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=False)
    tokenized_val_dataset = val_dataset.map(tokenizer_function, batched=False)
    
print(tokenized_train_dataset)
print(tokenized_val_dataset)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors='tf')

sample_data = [tokenized_train_dataset[i] for i in range(2)]
print(sample_data)
print(type(sample_data))
for data in sample_data : 
    data.pop('labels')
    data.pop('text')
print(sample_data)  
    
for data in data_collator(sample_data)["input_ids"] : 
    print(f"\n'>>> {tokenizer.decode(data)}'")
    


collated = data_collator(sample_data)
print(collated)


tf_train_set = tokenized_train_dataset.to_tf_dataset(
    columns = ["attention_mask", "input_ids", "labels"],
    shuffle=True, 
    batch_size = 4, 
    collate_fn=data_collator,
)

tf_val_set = tokenized_val_dataset.to_tf_dataset(
    columns = ["attention_mask", "input_ids", "labels"],
    shuffle=True, 
    batch_size = 4, 
    collate_fn=data_collator,
)

print(tf_train_set)
print(tf_val_set)

# Model Definition

import tensorflow as tf
from transformers import TFAutoModelForMaskedLM
from transformers import create_optimizer, AdamWeightDecay

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model = TFAutoModelForMaskedLM.from_pretrained(pretrained_transformer_model)

model.compile(optimizer=optimizer)
model.fit(x=tf_train_set, validation_data=tf_val_set, epochs=3)



