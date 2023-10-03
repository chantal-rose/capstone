from heapq import nlargest

from utils import get_embeddings, get_map, compute_text_similarity

class Ranker():
    def __init__(self):
        self.map = get_map()
        if(self.map==[]):
            raise ValueError("Model map is empty!! Please pre-compute it first!")

    def get_top_k_models(self, question: str, context: str, k: int=2) -> list[str]:
        embedding = get_embeddings(f"{question} {context}")
        best_models = nlargest(k, self.map, key=lambda m: compute_text_similarity(embedding, m['embeddings']))
        return [model['model_name'] for model in best_models]