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
    We have a relatively strict routine. For a few reasons. I just want to
    instill healthy sleeping habits for my kids, but also we want our kids
    to go to bed so that we can have a few hours to ourselves. Yeah. So at,
    or maybe 10 minutes before bedtime ... We’ll give them a countdown.
    We’ll say, "It’s 10 minutes before bedtime, five until bedtime, one
    minute.
    ''' , 

    '''
    Yeah. So yeah, knitting and yarn are definitely things that make me
    feel centered, and help me breathe again. I think part of that is the
    rhythm of working through a row of knitting. It really helps me to
    breathe and center myself, so that’s good. 
    ''' , 

    '''
    After she goes to bed, I need to do some housework and I also need to
    prepare for my job. Some work for my job too. I teach and I need to
    prepare the teaching materials... It’s because during the day I have to
    be with her and only after she went to bed or fall asleep, I can work.
    I can do whatever I need to do. . . I always tell myself that good for
    our health and you should go to bed early. Go to bed early, go to bed
    early, but every day I end up with bed... Yesterday, I went to bed at
    2:00am.
    ''' , 

    '''
    “We try as much as possible to keep our daily routine. So if we take a
    vacation, we still try to have quiet time and bedtime might be a little
    bit later, but we still follow everything.
    ''' , 
]


'''
2. Corpus Definitions
'''
max_corpus_size = int(1e4)


corpus_files = [
        'data/mommit.jsonl',
        'data/parenting.jsonl' , 
        'data/parents.jsonl' , 
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



