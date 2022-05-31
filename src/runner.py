from src.similarities.arora_similarity import AroraBeam
from src.experiments.domain_similarity import DomainSim
from src.similarities.sbert_similarity import SBERTSim

import argparse
import os
import json
import pandas as pd

parser = argparse.ArgumentParser(description='In this script')

parser.add_argument('--term_freq_path', required=False, default=False, type=str)
parser.add_argument('--embedding_path', required=False, default=False, type=str)
parser.add_argument('--community_json', required=False, default=False, type=str)
parser.add_argument('--target_communuity' , required=False, default=False, type=str)
parser.add_argument('--vocab_file')

parser.add_argument('--reddit_jsonl', required=False, default=False, type=str)
parser.add_argument('--body_field', required=False, default=False, type=str)
parser.add_argument('--id_field', required=False, default=False, type=str)

args = parser.parse_args()

if args.term_freq_path : 

    if (os.path.exists(args.term_freq_path) and os.path.exists(args.vocab_file)) : 
        print('Found term frequence matrix file. Loading...')
        df = pd.read_csv(args.term_freq_path)

    else : 

        print('Did not find term frequency path. Using community_json_file to build term_frequency_matrix.')
        assert os.path.exists(args.community_json, 'Community json required if term_frequency_path is invalid.')
        from src.experiments.counts import main

        
        






if args.embedding_path : 

    if os.path.exists(args.embedding_path) :

        print('Found pre-trained embeddings at : ', args.embedding_path)
        from gensim.models import KeyedVectors
        KeyedVectors.load(args.embedding_path)

    else : 

        print('No pre-trained embeddings found at  : ' , args.embedding_path, '. Training... ')

        from src.experiments import embeddings



