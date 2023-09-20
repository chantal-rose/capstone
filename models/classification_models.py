"""Model classes for Text Classification Task
FINANCE TEXT CLASSIFICATION (predicts bearish, bullish or neutral):
    text = "the USD has been trending lower"
    model = TextClassification("nickmuchi/deberta-v3-base-finetuned-finance-text-classification")
    model.predict(text)

FINANCE SENTIMENT (predicts negative, positive):
    text = "the USD has been trending lower"
    model = TextClassification("ProsusAI/finbert")
    model.predict(text)

SENTIMENT ANALYSIS:
    text = "I am happy"
    model = TextClassification("distilbert-base-uncased-finetuned-sst-2-english")
    model.predict(text)

TOXICITY CLASSIFIER:
    text = "You suck"
    model = TextClassification("s-nlp/roberta_toxicity_classifier")
    model.predict(text)

LANGUAGE DETECTION:
    text = "The end is near"
    model = TextClassification("papluca/xlm-roberta-base-language-detection")
    model.predict(text)

TOPIC CLASSIFICATION:
    text = '''Her mom had warned her. She had been warned time and again, but she had refused to believe her. She had done
     everything right, and she knew she would be rewarded for doing so with the promotion. So when the promotion was
      given to her main rival, it not only stung, it threw her belief system into disarray. It was her first big lesson
      in life, but not the last.'''
    model = TextClassification("alimazhar-110/website_classification")
    model.predict(text)

CRYPTOCURRENCY SENTIMENT ANALYSIS OF SOCIAL MEDIA:
    text = "Grayscale applies for new Ethereum futures ETF"
    model = TextClassification("ElKulako/cryptobert")
    model.predict(text)
"""
from base import HuggingFaceModel
from transformers import pipeline

TASK = "text-classification"


class TextClassification(HuggingFaceModel):
    def __init__(self, model_name):
        HuggingFaceModel.__init__(self)
        self.model_name = model_name
        self.pipeline = pipeline(TASK, model=self.model_name)

    def predict(self, text):
        response = self.pipeline(text)
        return response[0]["label"]
