from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import read_jsonl


class Corpus(object) : 

    def __init__(self, 
                corpus_definition, 
                maxlines=4e5, 
                vectorizer_definitions=[
                                {'max_features' : 1e4, 'vectorizer_type' : 'unigram' , 'ngram_range' : (1 , 1)},
                                {'max_features' : 1e5, 'vectorizer_type' : 'bigram' , 'ngram_range' : (2 , 2)},
                                {'max_features' : 1e5, 'vectorizer_type' : 'trigram' , 'ngram_range' : (3 , 3)}
                                ],
                content_field='body') -> None:

        self.corpus_definition = corpus_definition
        self.content_field = content_field
        self.maxlines = maxlines
        self.communities = list(corpus_definition.keys())
        self.num_communities = len(corpus_definition)
        self.vectorizer_definitions = vectorizer_definitions

        self.corpus = {}
        self.vectorizers = {}
        self.vocabs = {}
        self.counts = {}
        self.prob_word_given_domain = {}

        self._build_corpus()

        for vec_def in self.vectorizer_definitions :
            self._build_vectorizers(**vec_def)

        self.corpus_vocab, self.corpus_vocab_sizes = self._build_corpus_vocab()

        self._build_term_probability_matrix()

    def _build_corpus(self) -> None :

        print('Building the corpus...')

        for community, corpus in self.corpus_definition.items() : 
            self.corpus[community] = []            
            for subreddit in corpus : 
                self.corpus[community].extend(read_jsonl(subreddit['subreddit_path'], field=self.content_field, max_lines=self.maxlines))
                print('...Finished extracting posts from subreddit : r/{} belonging to the community : {}.'.format(subreddit, community))


    def _build_vectorizers(self, max_features, ngram_range=(1, 1), vectorizer_type='unigram') : 

        print('Preparing vectorizers, vocab and counts...')

        if vectorizer_type not in self.vectorizers : 
            self.vectorizers[vectorizer_type] = {}
            self.counts[vectorizer_type] = {}
            self.vocabs [vectorizer_type] = {}
            
        for community, data in self.corpus.items() : 

            vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
            X = vectorizer.fit_transform(data).toarray().sum(axis=0)
            total_count = np.sum(X)
            self.vectorizers[vectorizer_type][community] =   vectorizer
            self.counts[vectorizer_type][community] = {word:count for word,count in zip(vectorizer.get_feature_names(), X)}
            self.vocabs[vectorizer_type][community] = vectorizer.get_feature_names()
            # self.prob_word_given_domain[vectorizer_type][community] = {word:count/total_count for word,count in zip(vectorizer.get_feature_names(), X)}

            print('...Finished extracting {}s for {}.'.format(vectorizer_type , community))


    def _build_corpus_vocab(self) : 

        corpus_vocab = {}
        corpus_vocab_sizes = {}

        print('Building corpus vocab...')

        for vectorizer_type, community_vocabs in self.vocabs.items() :  

            print('Working on {}s...'.format(vectorizer_type))
            
            corpus_vocab[vectorizer_type] = []

            for community, vocab in community_vocabs.items() : 
                corpus_vocab[vectorizer_type] += vocab
                corpus_vocab[vectorizer_type] = list(set(corpus_vocab[vectorizer_type]))
                print('...Finished processing vocab for community {} ....'.format(community))

            corpus_vocab_sizes[vectorizer_type] = len(corpus_vocab[vectorizer_type])

        return corpus_vocab , corpus_vocab_sizes

    def _build_term_probability_matrix(self) : 

        term_prob_matrices = {}

        print('Building term probability matrix...')

        for vectorizer_type, vocab in self.corpus_vocab : 

            print('Working on {}s...'.format(vectorizer_type))

            term_prob_matrices[vectorizer_type] = np.zeros(self.corpus_vocab_sizes[vectorizer_type] , self.num_communities)

            for i in tqdm(range(self.corpus_vocab_sizes[vectorizer_type])) : 

                word = vocab[i]
                for j, community in enumerate(self.communities) : 

                    if word in self.counts[vectorizer_type][community][word] : 
                        term_prob_matrices[vectorizer_type][i , j] = self.counts[vectorizer_type][community][word]

            
            print(pd.DataFrame(term_prob_matrices[vectorizer_type], index=vocab, columns=self.communities))



if __name__ == '__main__' : 


    corpus = Corpus({'airbnb_hosts' : [{'subreddit' : 'airbnb_hosts' , 'subreddit_path' : 'data/airbnb_hosts.jsonl'}], 
                    'airbnb' : [{'subreddit' : 'airbnb' , 'subreddit_path' : 'data/airbnb.jsonl'}], 
                    'vrbo' : [{'subreddit' : 'vrbo' , 'subreddit_path' : 'data/vrbo.jsonl'}], 
                    'caloriecount' : [{'subreddit' : 'caloriecount' , 'subreddit_path' : 'data/caloriecount.jsonl'}],
                    'loseit' : [{'subreddit' : 'loseit' , 'subreddit_path' : 'data/loseit.jsonl'}],
                    })

    



            









        