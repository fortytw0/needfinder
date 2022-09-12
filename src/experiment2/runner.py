from src.experiment2.config import *
from src.experiment2.main import Experiment 
from src.experiment2.search import Search 

import os

if not os.path.exists(eval_results_dir) : 
    os.mkdir(eval_results_dir)


if MODE == 'EXPERIMENT' : 

    experiment = Experiment(max_corpus_size=max_corpus_size,
            corpus_files=corpus_files, 
            query_target_json=query_target_json_path, 
            results_dir=eval_results_dir)

    for model, model_params in experiment_definitions.items() :
        for model_param in model_params : 
            experiment.execute(model, model_param)

else : 

    search = Search(max_corpus_size=max_corpus_size, 
    corpus_files=corpus_files , 
    queries=queries,
    results_dir=eval_results_dir)


    for model, model_params in experiment_definitions.items() : 
        for model_param in model_params : 
            search.execute(model, model_param)


