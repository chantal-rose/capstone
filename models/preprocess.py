from utils import get_embeddings, get_map, compute_text_similarity

class Ranker():
    def __init__(self,question,context):
        self.question = question
        self.context = context
    def embed_question_context_pair(self):
        self.qustion_context_pair = self.question+self.context
        self.embeddings = get_embeddings(self.qustion_context_pair)
    def compute_model_similarity(self):
        self.map = get_map()
        self.similarity_values = []
        if(self.map==[]):
            print("Model map is empty!! Please pre-compute it first!")
            return
        for model in self.map:
            self.similarity_values.append((compute_text_similarity(self.embeddings,model["embeddings"]),model["model_name"]))
