import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from llm_utils import query_llm
import prompts


def _bool_qa_classifier(question: str, context: str, answer: str) -> torch.Tensor:
    reformulated_question = query_llm(messages=[
        {"role": "system", "content": prompts.REFORMULATE_QUERY},
        {"role": "user", "content": f"query: {question}\nanswer: {answer}"}
    ], model='gpt-3.5-turbo-1106')

    model_name = "PrimeQA/tydiqa-boolean-answer-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    with torch.inference_mode():
        res = tokenizer(reformulated_question, context, return_tensors="pt", truncation=True)
        probs = model(**res).logits.softmax(-1).squeeze()
    return probs


def get_generative_confidence(question: str, context: str, answer: str) -> float:
    probs = _bool_qa_classifier(question, context, answer)
    return probs[2].item()


def verify_answer(question: str, context: str, answer: str) -> bool:
    labels = [False, False, True]  # ['no', 'no-answer', 'yes']
    probs = _bool_qa_classifier(question, context, answer)
    label = labels[probs.argmax()]
    return label

def verify_answer_gpt(question: str, context: str, answer: str) -> bool:
    reformulated_question = query_llm(messages=[
        {"role": "system", "content": prompts.VERIFY_ANSWER},
        {"role": "user", "content": f"query: {question}\nanswer: {answer}\ncontext: {context}"}
    ], model='gpt-3.5-turbo-1106')
    return reformulated_question.lower().strip() == "true"