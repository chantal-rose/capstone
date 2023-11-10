
"""Model classes for Answer Verification
ANSWER VERIFICATION:
    text = "Is Ko Adang the biggest island in Tarutao National Marine Park?"
    context = "Ko Adang (Thai: เกาะอาดัง, pronounced [kɔ̀ʔ ʔāːdāŋ]) is the second biggest island within Tarutao National Marine Park, in Thailand, very close to Ko Lipe island. The island is 6 km long and 5 km wide. The highest point on the island is 690 m."
    model = PrimeQAAnswerVerificationModel()
    model.predict(question, context)


"""
from base import HuggingFaceModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PrimeQAAnswerVerificationModel(HuggingFaceModel):
    def __init__(self):
        HuggingFaceModel.__init__(self)
        self.model_name = "PrimeQA/tydiqa-boolean-answer-classifier"
        self.labels = [False, False, True]  # ['no', 'no-answer', 'yes']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, question, context):
        with torch.inference_mode():
            res = self.tokenizer(question, context, return_tensors="pt")
            label = self.labels[self.model(**res).logits.argmax()]
        return label