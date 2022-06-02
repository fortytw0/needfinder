from src.utils import read_json, create_dir_if_not_exist
from src.wordvectors import get_embedding_model
import os


class PreLoader(object) : 

    def __init__(self, 
                corpus_definition:dict=False, 
                corpus_definition_json:str=False,
                embedding:str='word2vec',
                word_vector_dir:str='data/wordvectors', 
                term_frequency_dir:str='data/term_frequency' , 
                community_words_dir:str='data/community_words', 
                force_train:bool=False) -> None:

        # Class Variables
        self.corpus_definition = self._load_corpus_definition(corpus_definition, corpus_definition_json)
        self.embedding = embedding
        self.force_train = force_train


        # Env Setup
        self.word_vector_dir = os.path.join(word_vector_dir , embedding)
        self.term_frequency_dir = term_frequency_dir
        self.community_words_dir = community_words_dir
        self._setup_env_dirs()

        # Vocabulary and Term Frequency setup


        # Word Vectors Setup
        self.communities_without_wordvectors = [community for community, exists in self._wordvectors_exist().items() if not exists ]
        self.embedding_model = get_embedding_model(self.embedding)



    def _load_corpus_definition(self, corpus_definition:dict=False, corpus_definition_json:str=False) -> None : 

        assert corpus_definition or corpus_definition, 'Provide either a corpus dictionary or a json with the corpus information.'
        if corpus_definition_json : corpus_definition = read_json(corpus_definition_json)
        return corpus_definition



    def _wordvectors_exist(self) -> dict : 
        
        exists = {}

        for community in self.corpus_definition.keys() : 
            
            if self.force_train : 
                exists[community] = False

            else : 
                wordvector_path = os.path.join(self.word_vector_dir, community)
                exists[community] = os.path.exists(wordvector_path)

        
        return exists

    def _setup_env_dirs(self) -> None:  

        for dir in [self.word_vector_dir, self.term_frequency_dir, self.community_words_dir] : 
            create_dir_if_not_exist(dir)


        


    def display_corpus_info(self, corpus_definition) : 

        print('Loaded JSON. Identified {} communities. The mappings are listed below : '.format(len(corpus_definition)))

        for community, subreddit in corpus_definition.items() : 
            print(community , ' : ' , subreddit)
            



