from src.corpus import Corpus
from src.wordvectors.word2vec import W2VEmbedding
from src.retriever import Retriever
from src.similarities.arora_similarity import AroraBeam
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import KeyedVectors


class AroraEncoder(object) : 

    def __init__(self, corpus:Corpus, embedding:W2VEmbedding, community, alpha=0.25, embedding_dimension=100) : 

        self.corpus = corpus
        self.community = community
        self.embedding = embedding.embeddings[self.community]
        self.vocab = self.corpus.vocabs['unigram'][self.community]
        self.alpha = alpha
        self.embedding_dim = embedding_dimension


    def fit(self, query_sentences) : 
        pass

    def _process_sentences(self, sentences) : 
        pass

    def _get_word_embedding(self, word) : 
        pass



