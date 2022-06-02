from src.wordvectors import word2vec

def get_embedding_model(embedding_name:str) : 

    if embedding_name == 'word2vec' : 
        return word2vec