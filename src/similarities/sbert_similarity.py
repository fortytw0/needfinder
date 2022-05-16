from src.similarities.similarity import Sim
from sentence_transformers import SentenceTransformer

class SBERTSim(Sim) : 

    def __init__(self, model_name:str='paraphrase-MiniLM-L3-v2') -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name)

    def build(self, jsonl_path: str, content_field: str) -> None:
        content = super().build(jsonl_path, content_field)
        self.matrix = self.model.encode(content)
        

if __name__ == '__main__' : 

    sbert_sim = SBERTSim()
    sbert_sim.build('data/airbnb_hosts.jsonl', 'body')
    print(sbert_sim.matrix.shape)