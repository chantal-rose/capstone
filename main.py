"""Main model for a pass through the system"""
# from flask import Flask, request
from llm_utils import GPT4InputParser,domain_label
from model_pipelines import get_answer_from_model
from model_pipelines import load_models
from model_pipelines import load_pipeline
from utils import filter_map
from utils import get_final_answer
from utils import get_top_k_models
from open_domain import get_context
from answer_verification import verify_answer


# app = Flask(__name__)
DOMAIN = "domain"
TYPE = "type"
K = 3


def send_input_to_system(models: dict, question: str, context: str) -> str:
    """Passes the user input to the system.

    This function implements the entire pipeline.
    NOTE: Consider moving the part after GPT4 parsing to a separate function for retries and feedback loop.

    :param models: Dictionary mapping models to their tokenizers and model classes
    :param user_input: Raw user query
    :return:
    """
    # parser = GPT4InputParser()
    # parser.parse(user_input)

    # type = parser.type
    # domain = parser.domain
    # question = parser.question
    # context = parser.context
    type = "extractive"
    

    if not context:
        context = get_context(question)
    domain = domain_label(context)

    top_k_embedding_models = get_top_k_models(question, context, K)
    top_k_domain_models = filter_map(DOMAIN, domain, K)
    top_k_type_models = filter_map(TYPE, type, K)

    answers = []
    answer_scores = []

    all_models = top_k_embedding_models + top_k_domain_models + top_k_type_models
    # TODO: Consider making it a set so that the same models aren't reinforcing the wrong answer

    for model in all_models:
        pipeline = load_pipeline(models, model)
        answer, confidence_score = get_answer_from_model(pipeline, models, model, question, context)
        if answer:
            answers.append(answer)
            answer_scores.append(confidence_score)

    temp_scores = [score for score in answer_scores if score is not None]
    average_score = sum(temp_scores) / len(temp_scores)
    answer_scores = [score if score is not None else average_score for score in answer_scores]

    final_answer = get_final_answer(answers, answer_scores)

    if not verify_answer(question, context, final_answer):
        final_answer += " (unknown)"
    print("Models picked:\n")
    all_models = [model["model_name"] for model in all_models]
    print(all_models)
    print("All returned answers:\n")
    print(answers)
    print("Final answer:\n")
    print(final_answer)
    return final_answer

    # TODO: implement feedback loop with retries and query reformulation
    # TODO: implement answer verification


# TODO: Create flask app endpoint
# @app.route("/ask", methods=['POST'])
# def ask_system():
#     request_data = request.get_json()
#
#     user_query = request_data["query"]
#     return send_input_to_system(model_map, user_query)


if __name__ == "__main__":
    model_map = load_models()
    query = ""

    send_input_to_system(model_map, query)
    # app.run(port="8082", threaded=True, host=("0.0.0.0"))
