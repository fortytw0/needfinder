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
    The reason I don’t do it [physical activity] is because I’m on my own, my
    carer comes in the morning and the night and I’m with my fiancée or
    whatever most of the day so to get into the pool and get changed to
    exercise you need someone there, so I like to be as independent as possible and that’s probably why I don’t exercise and do that stuff in my
    own time. [...] With gaming I can just have my PC set up on sleep and
    roll up to it and use it and we are good to go.
    ''' , 

    '''
    One of the reasons
    I don’t play [movement-based games] is it involves a lot of set up
    depending on what game you want to play
    ''' , 

    '''
    if the building has accessibility but
    it requires someone to operate the lift for you, you don’t feel very
    independent, you don’t want to feel as if people are going out of their
    way. 
    ''' , 

    '''
    the main problem is like most wheelchair users in England,
    they’re told by doctors [...] this is kind of the end, like once you’re
    in a wheelchair that’s sort of the of it [physical activity] [...] so it’s
    difficult to find a large enough group
    ''' , 

    '''
    I don’t like getting called
    out. There can be a lot of rage like you can make a mistake and die
    first [...] and people might not know you’re disabled, like you die on
    an easy part where everyone lived. [...] I don’t want to have to explain
    myself
    ''' , 

    '''
    So in Tell-Tale games, they say at the end of the chapter ’66
    percent of people made this decision’, and you can compare yourself
    so if you’re climbing Everest you can give them options to go this way,
    or tie yourself to this tree, so this influences PA that the person does
    [...]." (Simon) - "Like Facebook games...it will come up on your feed
    showing you how well others have done, giving people a challenge,
    like global challenges so that’s involving others, but it doesn’t really
    affect you
    ''' , 

    '''
    I think that an issue that a lot of wheelchair users have with
    games, because obviously there aren’t many they seem a bit handholdy sort of alienated in a way, it feels very specifically designed for
    wheelchair users while someone in a wheelchair could use it, when it’s
    simple it seems obvious that it’s for someone in a wheelchair
    ''' , 

    '''
    game is there for me to escape and forget what I’m
    doing [...] it allows me not to be reminded of what I look like
    '''

    '''
    you don’t think about doing these activities when you’re playing
    the game like collecting these pizzas or fighting robots, so it becomes
    one more robot, one more robot, and you don’t even realise you’ve
    done PA
    '''
]


'''
2. Corpus Definitions
'''
max_corpus_size = int(1e4)


corpus_files = [
        'data/disability.jsonl',
        'data/disabledgamers.jsonl',
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



