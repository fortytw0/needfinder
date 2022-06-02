

class Retriever(object) : 

    def __init__(self) -> None:
        pass

    def train(self, unigrams:bool=True, bigrams:bool=False) -> None : 
        pass

    def get_community_keywords(self, sentence:str, community_name:str, threshold:float=0.5) -> list(str) : 
        pass

    def get_similar_quotes(self, query_sentence:str, corpus:list) -> tuple(list(str), list(str), list(str)) : 
        pass

    def _read_json_file(self, json_path:str) -> None : 
        pass

    def _train_word2vec(self) -> None : 
        pass

    def _train_community_term_matrix(self) -> None : 
        pass


        
        
    