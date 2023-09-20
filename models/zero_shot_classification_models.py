
"""Model classes for Zero Shot Classification Task
DEBERTA TASKSOURCE NLI:
    text = "one day I will see the world"
    candidate_labels = ['travel', 'cooking', 'dancing']
    model = ZeroShotClassificationModel("sileod/deberta-v3-base-tasksource-nli")
    model.predict(text, candidate_labels)


"""
from base import HuggingFaceModel
from transformers import pipeline

TASK = "zero-shot-classification"


class ZeroShotClassificationModel(HuggingFaceModel):
    def __init__(self, model_name):
        HuggingFaceModel.__init__(self)
        self.model_name = model_name
        self.pipeline = pipeline(TASK, model=self.model_name)

    def predict(self, text, candidate_labels):
        response = self.pipeline(text, candidate_labels)
        zipped = zip(response["labels"], response["scores"])
        sorted_labels = sorted(zipped, key=lambda x: x[1], reverse=True)
        return sorted_labels[0]
