'''
1. User Defined Variables
'''

MODE = 'SEARCH'
query_target_json_path = 'annotations/query_targets.json'
eval_results_dir = 'data/results/experiment2/'


'''
1a. Query Definitions : Only required for search
'''

queries = [
    '''
    “By using the pump buttons of my CC cream, it allows
    me to easily track the exact amount I have applied or need to apply
    ''' , 

    '''
    I use smell to distinguish makeup products with diferent
    scents, especially when they have similar physical packaging, such as
    lipsticks
    ''' , 

    '''
    I normally
    would have someone to do eyeliner for me because I cannot do it very
    well myself 
    ''' , 

    '''
    “I am legally blind and can see
    color contours with magnifers, I often use the huge 10x magnifcation mirrors for help with knowing colors and contours while doing
    makeup and checking after makeup
    ''' , 
]


'''
2. Corpus Definitions
'''
max_corpus_size = int(1e4)


corpus_files = [
        'data/makeupaddiction.jsonl',
        'data/blind.jsonl' , 
        'data/makeup.jsonl' , 
        ]


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
        },
        
        ] , 

        'SBERT' : [

        {
            'model_name' : 'paraphrase-MiniLM-L3-v2' , 
            'experiment_name' : 'paraphrase',
            'sim' : 'cos'
        } , 
        {
            'model_name' : 'multi-qa-mpnet-base-dot-v1' , 
            'experiment_name' : 'multi-qa-mpnet',
            'sim' : 'dot'

        }
        ]  , 

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
    }

 


experiment_definitions = {
    
    'SBERT' : [
        {
            'model_name' : 'all-mpnet-base-v2' , 
            'experiment_name' : 'all-mpnet-base-v2',
            'sim' : 'cos'
        } , 

    ]
}



