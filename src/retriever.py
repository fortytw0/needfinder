from nltk.data import retrieve
from src.corpus import Corpus
from src.wordvectors.word2vec import Word2Vec

import re
from tqdm import tqdm

class Retriever(object) : 

    def __init__(self, corpus:Corpus, community:str, threshold:float=0.5) -> None:
        self.corpus = corpus
        self.community = community
        self.threshold = threshold

        self.community_words = self._build_community_words()

    def retrieve(self, query_sentences, match_criteria=0.5, topk=10) -> None : 
        
        print('Identifying community words in query sentences...')
        queryCW = self._identify_community_words(query_sentences)

        print('Identifying community words in the corpus...')
        corpusCW = self._identify_community_words(self.corpus.corpus[self.community])

        print('Matching based on community words...')
        matches = self._match_based_on_CW(queryCW, corpusCW)
        

        print('Preparing retrieved object...')
        retrived_object = {}
        for query_index, corpus_indices in tqdm(matches.items()) : 
            retrived_object[query_sentences[query_index]] = []

            for c in corpus_indices : 
                retrived_object[query_index].append({'reddit_post':self.corpus.corpus[c['index']] , 
                                                'matched_words':c['matched_words']})

        print('...Finished preparing retrieved object')
        return retrived_object

        

    def _match_based_on_CW(queryCW, corpusCW) : 
        '''
        TODO : Match Criteria not implemented. Top-K not implemented.
        '''

        matches = {}

        for i, qcw in tqdm(enumerate(queryCW)) : 
            matches[i] = []

            for j, ccw in enumerate(corpusCW) :
                if set(qcw).intersection(set(ccw)) : 
                    matches[i].append({'index' : j, 
                                    'matched_words' : list(set(qcw).intersection(set(ccw)))})

        return matches

    def _build_community_words(self) -> list(str) : 

        community_words = self.corpus['unigram'][self.community]
        community_words = community_words[community_words>self.threshold].to_dict()
        return community_words

        

    def _identify_community_words(self, sentences:str) -> tuple(list(str), list(str), list(str)) : 
        sentences = self._process_sentences(sentences)
        community_words = []

        for sentence in sentences : 
            community_words.append([w for w in sentence if w in self.community_words])

        return community_words

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

        

if __name__ == '__main__' : 

    corpus = Corpus({'airbnb_hosts' : [{'subreddit' : 'airbnb_hosts' , 'subreddit_path' : 'data/airbnb_hosts.jsonl'}], 
                    'airbnb' : [{'subreddit' : 'airbnb' , 'subreddit_path' : 'data/airbnb.jsonl'}], 
                    'vrbo' : [{'subreddit' : 'vrbo' , 'subreddit_path' : 'data/vrbo.jsonl'}], 
                    'caloriecount' : [{'subreddit' : 'caloriecount' , 'subreddit_path' : 'data/caloriecount.jsonl'}],
                    'loseit' : [{'subreddit' : 'loseit' , 'subreddit_path' : 'data/loseit.jsonl'}],
                    })

    retriever = Retriever(corpus, 'airbnb_hosts' , 0.4)

    

        
    