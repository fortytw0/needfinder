import numpy as np

import os
from tqdm import tqdm
 

class GloveEmbedding(object) : 

    def __init__(self, 
                save_dir:str, 
                dimension:int=100) : 
               

        self.save_dir = save_dir
        self.dimension = dimension
        self.embedding_basename = 'glove.6B.{}d.txt'.format(self.dimension)
        self.embedding_path = os.path.join(self.save_dir , self.embedding_basename)


        if not os.path.exists(self.embedding_path) :
            raise FileNotFoundError('Could not find the GloVe embeddings.' 
            'Please download them from https://nlp.stanford.edu/data/glove.6B.zip,'
            'and place the decompressed conents in {}'.format(self.save_dir))

        self.embeddings = self._build_embedding()

    def _build_embedding(self) : 

        word_embeddings = {}

        with open (self.embedding_path) as f : 
            print('Loading GLOVE embedding from : ' , self.embedding_path)
            for line in tqdm(f.readlines()) : 
                word, embedding = line.split(' ' , 1)
                embedding = np.fromstring(embedding , sep=' ')
                word_embeddings[word] = embedding

        word_embeddings['<UNK>'] = np.mean(list(word_embeddings.values()), axis=0)

        return word_embeddings

        
 
