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
                retrain=False, 
                dimension=100, 
                alpha=1e4) -> None:

        self.save_dir = savedir
        self.corpus = corpus
        self.run_train = retrain

        self.dimension = dimension
        self.alpha = alpha
        self.embeddings = {}
        self.extension = '.wordvector'

        self.load()

    def _process_sentences(self, sentences, vocab) : 

        processed_sentences = []
        num_sentences = len(sentences)

        for i in tqdm(range(num_sentences)) : 
            sentence = sentences[i].strip().lower()
            tokens = re.findall(r'(\w+)' , sentence)
            tokens = [t for t in tokens if len(t) > 1]
            tokens = [t if t in vocab else '<UNK>' for t in tokens]
            processed_sentences.append(tokens)

        return processed_sentences

    def _train_model(self, processed_sentences, vocab, num_epochs=100) : 

        num_processed_sentences = len(processed_sentences)
        num_vocab = len(vocab)

        model = Word2Vec(vector_size=self.dimension, 
                            alpha=self.alpha)

        model.build_vocab(processed_sentences, update=False)
        model.train(processed_sentences, 
                    total_examples=num_processed_sentences, 
                    total_words=num_vocab, 
                    epochs=num_epochs)

        return model

    def _train_community_embeddings(self, community, num_epochs=100) : 

        print('Working on {}...'.format(community))
        print('Processing sentences...')

        processed_sentences = self._process_sentences(self.corpus.corpus[community], self.corpus.vocabs['unigram'][community])
        
        print('...Finished processing sentences')
        print('Training word2vec model...')

        model = self._train_model(processed_sentences, self.corpus.vocabs['unigram'][community], num_epochs=num_epochs)

        print('...Finished training model')
        print('Saving word2vec model...')
        
        model.wv.save(os.path.join(self.save_dir , 'word2vec' , community + self.extension))
        self.embeddings[community] = model.wv

        print('Finished training and saving word embedding for community {} ...'.format(community))

    def _train_all_community_embeddings(self) : 

        for community in self.corpus.communities : 
            self._train_community_embeddings(community)

            

    def load(self) : 

        print('Loading word2vec...')

        if self.run_train :
            print('Train flag is True, running training...')
            self._train_all_community_embeddings()

        else :
            print('Checking if all communities have embeddings...')

            if not os.path.isdir(self.save_dir) : 
                raise FileNotFoundError('Your specified save directory could not be found : ' , self.save_dir)

            if not create_dir_if_not_exist(os.path.join(self.save_dir, 'word2vec')) :  # create_dir_if_not_exist creates word2vec model if it does not exists and returns False
                self._train_all_community_embeddings() # since create_dir_if_not_exist returns False, that means the embeddings need to be trained

            for community in self.corpus.communities : 

                wv_fpath = os.path.join(self.save_dir , 'word2vec' , community + self.extension)
                print('Checking for word_vectors at : ' , wv_fpath)

                if os.path.exists(wv_fpath) : 
                    print('{} community already has a wordvector file, which will be loaded...'.format(community))
                    self.embeddings[community] = KeyedVectors.load(wv_fpath)
                else : 
                    print('{} community does not have a wordvector file, hence will be trained...'.format(community))
                    self._train_community_embeddings(community)
                    

if __name__ == '__main__' : 

    corpus = Corpus({'airbnb_hosts' : [{'subreddit' : 'airbnb_hosts' , 'subreddit_path' : 'data/airbnb_hosts.jsonl'}], 
                    'airbnb' : [{'subreddit' : 'airbnb' , 'subreddit_path' : 'data/airbnb.jsonl'}], 
                    'vrbo' : [{'subreddit' : 'vrbo' , 'subreddit_path' : 'data/vrbo.jsonl'}], 
                    'caloriecount' : [{'subreddit' : 'caloriecount' , 'subreddit_path' : 'data/caloriecount.jsonl'}],
                    'loseit' : [{'subreddit' : 'loseit' , 'subreddit_path' : 'data/loseit.jsonl'}],
                    })
    print((corpus.corpus.keys()))
    print(corpus.vocabs.keys())

    embedding = W2VEmbedding(corpus, 'data/wordvectors')




    