import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from llm_utils import query_llm
import prompts


def verify_answer(question: str, context: str, answer: str):
    reformulated_question = query_llm(messages=[
        {"role": "system", "content": prompts.REFORMULATE_QUERY},
        {"role": "user", "content": f"query: {question}\nanswer: {answer}"}
    ])

    model_name = "PrimeQA/tydiqa-boolean-answer-classifier"
    labels = [False, False, True]  # ['no', 'no-answer', 'yes']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    with torch.inference_mode():
        res = tokenizer(reformulated_question, context, return_tensors="pt")
        label = labels[model(**res).logits.argmax()]
    return label