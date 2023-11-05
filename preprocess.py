from heapq import nlargest

from utils import get_embeddings, get_map, compute_similarity_between_embeddings


class Ranker:
    def __init__(self):
        self.map = get_map()
        if not self.map:
            raise ValueError("Model map is empty!! Please pre-compute it first!")

    def get_top_k_models(self, question: str, context: str, k: int = 2) -> list[str]:
        embedding = get_embeddings(f"{question} {context}")
        best_models = nlargest(k, self.map,
                               key=lambda m: compute_similarity_between_embeddings(embedding, m['embeddings']))
        return [(model['model_name'], compute_similarity_between_embeddings(embedding, model['embeddings'])) for model in best_models]

if __name__ == "__main__":
    r = Ranker()
    question = "The double-strand breaks occur along the DNA backbone. Describe the process by which the breaks occur"
    context = "During meiosis, double-strand breaks occur in chromatids. The breaks are either repaired by the exchange of genetic material between homologous nonsister chromatids, which is the process known as crossing over (Figure 1A), or they are simply repaired without any crossing over (Figure 1B). Plant breeders developing new varieties of corn are interested in determining whether, in corn, a correlation exists between the number of meiotic double-strand chromatid breaks and the number of crossovers."
    print(r.get_top_k_models(question, context, k=10))