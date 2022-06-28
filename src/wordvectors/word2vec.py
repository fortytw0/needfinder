from typing import List
from gensim.models import KeyedVectors, Word2Vec
from tqdm import tqdm
import re
import os

from src.corpus import Corpus
from src.utils import create_dir_if_not_exist


class W2VEmbedding(object) : 

    def __init__(self,   
                corpus:Corpus,
                savedir:str,
                community:str,
                retrain=False, 
                dimension=100, 
                alpha=1e-4) -> None:

        self.save_dir = savedir
        self.corpus = corpus
        self.run_train = retrain
        self.community = community
        self.dimension = dimension
        self.alpha = alpha
        
        self.extension = '.wordvector'


        self.embeddings = self.load()
        

    def _process_sentences(self, sentences:List[str]) : 
        
        processed_sentences = []

        for sentence in tqdm(sentences) : 
            sentence = sentence.strip().lower()
            tokens = re.findall(r'(\w+)' , sentence)
            tokens = [t for t in tokens if len(t) > 1]
            tokens = [t if t in self.corpus.vocab else '<UNK>' for t in tokens]
            processed_sentences.append(tokens)

        return processed_sentences

    def _train_model(self, processed_sentences:List[str], num_epochs:int=100) : 

        num_processed_sentences = len(processed_sentences)

        model = Word2Vec(vector_size=self.dimension, 
                            alpha=self.alpha)

        model.build_vocab(processed_sentences, update=False)
        model.train(processed_sentences, 
                    total_examples=num_processed_sentences, 
                    total_words=self.corpus.vocab_size, 
                    epochs=num_epochs)

        return model

    def _train_community_embeddings(self, num_epochs:int=100) : 

        print('Training embeddings for {self.community} community...')
        print('Processing sentences...')

        processed_sentences = self._process_sentences(self.corpus.data)
        
        print('...Finished processing sentences')
        print('Training word2vec model...')

        model = self._train_model(processed_sentences, num_epochs=num_epochs)

        print('...Finished training model')
        print('Saving word2vec model...')
        
        model.wv.save(os.path.join(self.save_dir , 'word2vec' , self.community + self.extension))
        print('Finished training and saving word embedding for community {} ...'.format(self.community))

        return model.wv
        

            

    def load(self) : 

        print('Loading word2vec...')

        if self.run_train :
            print('Train flag is True, running training...')
            return self._train_community_embeddings()

        else :
            
            # Checking if the directory exists/contains pre-trained embeddings. 
            # Only train if the directory does not exist, or the community does 
            # not have previously trained embeddings. 

            if not os.path.exists(self.save_dir) : 
                print('Your specified save directory could not be found : ' , self.save_dir)
                create_dir_if_not_exist(self.save_dir)
                print('Created save directory.')

            if not create_dir_if_not_exist(os.path.join(self.save_dir, 'word2vec')) :  # create_dir_if_not_exist creates word2vec folder if it does not exists and returns False
                return self._train_community_embeddings() # since create_dir_if_not_exist returns False, that means the embeddings need to be trained
            

            wv_fpath = os.path.join(self.save_dir , 'word2vec' , self.community + self.extension)
            print('Checking for word_vectors at : ' , wv_fpath)

            if os.path.exists(wv_fpath) : 
                print('{} community already has a wordvector file, which will be loaded...'.format(self.community))
                return KeyedVectors.load(wv_fpath)
            else : 
                print('{} community does not have a wordvector file, hence will be trained...'.format(self.community))
                return self._train_community_embeddings()
                    

if __name__ == '__main__' : 

    corpus = Corpus(['data/airbnb_hosts.jsonl'])
    embedding = W2VEmbedding(corpus, savedir='data/wordvectors', community='airbnb_hosts' )
    print(embedding['<UNK>'])
