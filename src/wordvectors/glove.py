import numpy as np

from src.corpus import Corpus
from tqdm import tqdm

class GloveEmbedding(object) : 

    def __init__(self, 
                corpus:Corpus,
                embedding_path:str, 
                dimension:int=100) : 
               

        self.corpus = corpus
        self.embedding_path = embedding_path
        self.dimension = dimension
        self.embeddings = self._build_embedding()

    def _build_embedding(self) : 

        word_embeddings = {}

        with open (self.embedding_path) as f : 
            print('Loading embedding from : ' , self.embedding_path)
            for line in tqdm(f.readlines()) : 
                word, embedding = line.split(' ' , 1)
                embedding = np.fromstring(embedding , sep=' ')
                word_embeddings[word] = embedding

        word_embeddings['<UNK>'] = np.mean(list(word_embeddings.values()), axis=0)

        return word_embeddings

        
 
