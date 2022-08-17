'''
1. User Defined Variables
'''
query_target_json_path = 'annotations/query_targets.json'
eval_results_dir = 'data/results/experiment2/'

'''
2. Corpus Definitions
'''
max_corpus_size = int(1e4)


corpus_files = [
        'data/airbnb_hosts.jsonl',
        'data/caloriecount.jsonl',
        'data/vrbo.jsonl']


'''
2. Experiment Definitions
'''

old_experiments = {'BERT' : [
        {
            'model_name' : 'bert-base-uncased' , 
            'max_length' : 512 , 
            'experiment_name' : 'bert-uncased-512'
        }, 
        {
            'model_name' : 'bert-base-uncased' , 
            'max_length' : 256 , 
            'experiment_name' : 'bert-uncased-256'
        },
        {
            'model_name' : 'bert-base-uncased' , 
            'max_length' : 128 , 
            'experiment_name' : 'bert-uncased-128'
        }
        ]  
                  
        }

 


experiment_definitions = {
   
    'Longformer' : [
        {
            'model_name' : 'allenai/longformer-base-4096' , 
            'tokenizer_name' : 'roberta-base' , 
            'max_length' : 1028 , 
            'experiment_name' : 'longformer-1028'
        } , 
        {
            'model_name' : 'allenai/longformer-base-4096' , 
            'tokenizer_name' : 'roberta-base' , 
            'max_length' : 512 , 
            'experiment_name' : 'longformer-512'
        }
    ] , 
    'SBERT' : [
        {
            'model_name' : 'all-mpnet-base-v2' , 
            'experiment_name' : 'all-mpnet'
        } , 
        {
            'model_name' : 'paraphrase-MiniLM-L3-v2' , 
            'experiment_name' : 'paraphrase'
        } , 
        {
            'model_name' : 'multi-qa-mpnet-base-dot-v1' , 
            'experiment_name' : 'multi-qa-mpnet'
        }
    ]
}



