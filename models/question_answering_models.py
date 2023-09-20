"""Model classes for Question Answering Task
BERT LONGFORMER:
    question = "When did Beyonce start becoming popular?"
    context = '''Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American
    singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various
    singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group
     Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl
      groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which
      established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100
      number-one singles \"Crazy in Love\" and \"Baby Boy\"")'''
    model = ExtractiveQuestionAnsweringModel("mrm8488/longformer-base-4096-finetuned-squadv2")
    model.predict(question, context)


"""
from base import HuggingFaceModel
from transformers import pipeline

TASK = "question-answering"


class ExtractiveQuestionAnsweringModel(HuggingFaceModel):
    def __init__(self, model_name):
        HuggingFaceModel.__init__(self)
        self.model_name = model_name
        self.pipeline = pipeline(TASK, model=self.model_name)

    def predict(self, question, context, **kwargs):
        response = self.pipeline(question=question, context=context)
        return response[0]["answer"]
