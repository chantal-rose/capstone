import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from llm_utils import query_llm
import prompts


def verify_answer(question: str, context: str, answer: str) -> bool:
    reformulated_question = query_llm(messages=[
        {"role": "system", "content": prompts.REFORMULATE_QUERY},
        {"role": "user", "content": f"query: {question}\nanswer: {answer}"}
    ], model='gpt-3.5-turbo-1106')

    labels = [False, False, True]  # ['no', 'no-answer', 'yes']
    model_name = "PrimeQA/tydiqa-boolean-answer-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    with torch.inference_mode():
        res = tokenizer(reformulated_question, context, return_tensors="pt", truncation=True)
        probs = model(**res).logits.softmax(-1).squeeze()

    return labels[probs.argmax()]


def verify_answer_gpt(question: str, context: str, answer: str) -> bool:
    reformulated_question = query_llm(messages=[
        {"role": "system", "content": prompts.VERIFY_ANSWER},
        {"role": "user", "content": f"query: {question}\nanswer: {answer}\ncontext: {context}"}
    ], model='gpt-3.5-turbo-1106')
    return reformulated_question.lower().strip() == "true"