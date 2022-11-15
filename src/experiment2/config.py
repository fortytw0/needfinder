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
    Entrepreneurs might’ve forgotten the class material by the time they have that need, especially
    if it’s not required, and then they won’t attend a session until the
    need is there.
    ''' , 

    '''
    I kept calling and he went on FaceTime and he went through
    all the settings and he told me what to punch in
    ''' , 

    '''
    I was held hostage at one point with my website. I was
    so frustrated and it brought tears because that’s how I sell my products. And for probably close to three months, my website was down
    ''' , 
]


'''
2. Corpus Definitions
'''
max_corpus_size = int(1e4)


corpus_files = [
        'data/local_entrepreneurs.txt',
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



