from gensim.models import KeyedVectors
from src.similarities.similarity import Sim
from src.wordvectors.word2vec import W2VEmbedding

from src.corpus import Corpus


import time
from tqdm import tqdm
import numpy as np
import re
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


class AroraBeam(object) : 


    def __init__(self, embedding:W2VEmbedding, 
                corpus:Corpus,  
                alpha:float=1e-4,  
                embedding_dimension:int=50, 
                eager_load:bool=False) -> None:
        
        # User Defined
        self.alpha = alpha
        self.corpus = corpus
        self.word_embedding = embedding.embeddings
        self.embedding_dimension = embedding_dimension      

        

        if eager_load : 
            self.corpus_repr = self._fit_corpus()
        else : 
            self.corpus_repr = None
        

    
    def _fit_corpus(self) -> np.array : 
        print("Preparing Corpus embedding repr...")
        return self._fit(self.corpus.data) 


    def _fit(self, texts: list) -> np.array:
        text_repr = []
        for text in tqdm(texts) : 
            sentence_embedding = self._get_sentence_embedding(text)
            text_repr.append(sentence_embedding)
        return np.array(text_repr)

    def rank(self, queries) -> float:
        print('Preparing queries embedding repr... ')
        text_repr = self._fit(queries)
        
        if self.corpus_repr == None :
            self.corpus_repr = self._fit_corpus()
            
            
        print('Corpus_REPR.shape' , self.corpus_repr.shape)
        print('Query_REPR.shape' , text_repr.shape)
        return cosine_similarity(self.corpus_repr, text_repr)


    def word_wise_rank(self, quote:str, post:str) : 

        quote = self._process_sentence(quote)
        post = self._process_sentence(post)

        quote_vectors = []
        post_vectors = []

        quote_labels = []
        post_labels = []

        for word in quote : 
            word_vector = self._get_word_embedding(word)
            if (word in self.corpus.vocab) and (word_vector is not None) : 
                smoothing_factor = self.alpha / (self.alpha + self.corpus.word_probs[word])
                smoothing_factor = round(smoothing_factor, 2)
                quote_labels.append(word + '_{}'.format(smoothing_factor))
                quote_vectors.append(word_vector)

        for word in post : 
            word_vector = self._get_word_embedding(word)
            if (word in self.corpus.vocab) and (word_vector is not None) : 
                smoothing_factor = self.alpha / (self.alpha + self.corpus.word_probs[word])
                smoothing_factor = round(smoothing_factor, 2)
                post_labels.append(word + '_{}'.format(smoothing_factor))
                post_vectors.append(word_vector)

        return post_labels, post_vectors, quote_labels, quote_vectors 


    def _get_sentence_embedding(self, sentence:str) -> None : 

        num_unks = 0
        sentence = self._process_sentence(sentence)
        if len(sentence) == 0 : 
            return np.zeros((self.embedding_dimension,))

        sum = np.zeros((self.embedding_dimension,))

        for word in sentence : 
            
            word_vector = self._get_word_embedding(word)

            try :    
                
                if (word in self.corpus.vocab) and (word_vector is not None) : 
                    smoothing_factor = self.alpha / (self.alpha + self.corpus.word_probs[word])
                    sum+=smoothing_factor*word_vector

                else : 
                    num_unks += 1

            except Exception as e :

                print(e)
                print(self._get_word_embedding(word))
                print('Word : ' , word)

        sentence_embedding = sum/len(sentence)        
        return sentence_embedding


    def _get_context_embedding(self, context:str) -> None : 

        context_matrix = np.zeros((self.embedding_dimension , len(context)))
        
        for i, sentence in enumerate(context) : 
            context_matrix[: , i] = self._get_sentence_embedding(sentence)

        U, S, VT = svds(context_matrix, k=1)

        for i, sentence in enumerate(context) : 
                context_matrix[: , i] -= U@U.T@context_matrix[: , i]

        return context_matrix

    def _get_word_embedding(self, word:str) -> np.array : 
            
        try :
            return self.word_embedding[word]
        except : 
            return None


    def _process_sentence(self, sentence) : 
        
        sentence = sentence.lower().strip()
        tokens = re.findall(r'(\w+)' , sentence)
        tokens = [t for t in tokens if len(t) > 1]
        tokens = [t if t in self.corpus.vocab else '<UNK>' for t in tokens]

        return tokens


if __name__ == '__main__' : 

    corpus = Corpus(['data/airbnb_hosts.jsonl'])

    embedding = W2VEmbedding(corpus, savedir='data/wordvectors', community='airbnb_hosts' )

    
    ab = AroraBeam(embedding=embedding, 
                    corpus=corpus, 
                    embedding_dimension=100)

    import json
    with open('data/labels.json') as f: 
        quotes = json.load(f)[1]['quotes']

    # sim = ab.rank(quotes)

    sample_quote = quotes[0]
    sample_post = '''
    &gt; Sure but one block can be very different from another block in terms the renter's experience -- as I wrote being in a unit on a quiet residential street is very different than being in a unit in between a bar and drug store -- and both of those units could be within a circle on a map.

this is simply not accurate at all. Yes this is true for LOCALS where they would know "oh this street is sketchy" and what have you. 

But expecting tourist to know what street is what is just naive.

&gt;Why? And even if it really is a huge security issue then AirBnB could set it up so that only "vetted" renters can see it, or only renters with a review score above a certain threshold.

what is a "vetted" renter? just because you got 5*s before, doesn't mean you are trust worthy when you are anonymous. especially since there is no way to know if you were the one that broke into my house or not.

&gt;I suspect the top 2 reasons that AirBnB hides it are:

you are wrong already. hosts hide it, not airbnb. Airbnb default is to show it.


&gt;They don't want people bypassing AirBnB and renting the unit directly with the host (same reason they hide the host's contact info on pages prior to booking).

This is just plain reach. the address alone doesnt magically mean I can contact the owner to set up off site payments. more over any host that randomly accepts offsite payments are idiots themselves.

most hosts thatDO have off site booking (via their own site) usually has their branding info right on their description. a quick google will get you this info. as well as those that have offsite bookings will likely enable `show full address`

&gt;In cities where AirBnBs are currently not legal it makes it harder for people to see if their neighbor is violating the law.

Also not even remotely accurate. you already know the house by pictures AND the general location of the house. if you dont know the look of your neighbors houses....you gotta get out more.



Whats more plausible? Airbnb hosts hiding for security reasons (thieves can see when a house is "empty")

or whatever thoughts you just spewed that makes next to no sense?
    '''

    post_labels, post_vectors, quote_labels, quote_vectors = ab.word_wise_rank(sample_quote, sample_post)
    sim = cosine_similarity(post_vectors, quote_vectors)

    import pandas as pd 

    df = pd.DataFrame(sim, index=post_labels, columns=quote_labels)
    df.to_csv('data/results/arora_sim_wordwise_june21.csv')

    
            
            
            


    



    

    

