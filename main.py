"""Main model for a pass through the system"""
# from flask import Flask, request
import pandas as pd

from model_pipelines import get_answer_from_model
from model_pipelines import load_models
from model_pipelines import load_model
from model_pipelines import load_pipeline
from open_domain import get_context
from utils import filter_map
from utils import get_final_answer
from utils import get_top_k_models


# app = Flask(__name__)
DOMAIN = "domain"
TYPE = "type"
K = 2


def send_input_to_system(models: dict, question: str, context: str, domain_test: str) -> str:
    """Passes the user input to the system.

    This function implements the entire pipeline.
    NOTE: Consider moving the part after GPT4 parsing to a separate function for retries and feedback loop.

    :param models: Dictionary mapping models to their tokenizers and model classes
    :param user_input: Raw user query
    :return:
    """
    type = "extractive"

    if not context:
        context = get_context(question)
    domain = domain_test

    top_k_domain_models = filter_map(DOMAIN, domain, K)
    top_k_embedding_models = get_top_k_models(question, context, K, top_k_domain_models)
    top_k_type_models = filter_map(TYPE, type, K, top_k_embedding_models + top_k_domain_models)

    answers = []
    answer_scores = []
    final_models = []

    all_models = top_k_embedding_models + top_k_domain_models + top_k_type_models
    
    all_model_names = [model["model_name"] for model in all_models]
    if len(all_model_names) != len(set(all_model_names)):
        print("Not equal")
        breakpoint()
    
    # TODO: Consider making it a set so that the same models aren"t reinforcing the wrong answer

    for model in all_models:
        models = load_model(model["model_name"])

        pipeline = load_pipeline(models, model)
        try:
            print("Model: ", model["model_name"])
            answer = get_answer_from_model(pipeline, models, model, question, context)
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Exception for", model["model_name"])
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)
            continue
        else:
            if answer:
                final_models.append(model["model_name"])
                answers.append(answer)

    final_answer = get_final_answer(answers)

    # if not verify_answer(question, context, final_answer):
    #     final_answer += " (unknown)"
    print("Question: ", question)
    df = pd.DataFrame(list(zip(final_models, answers, answer_scores)),
               columns =["Model", "Answer", "Score"])
    print(df)
    print("Final answer:\n")
    print(final_answer)
    return final_answer

    # TODO: implement feedback loop with retries and query reformulation
    # TODO: implement answer verification


# TODO: Create flask app endpoint
# @app.route("/ask", methods=["POST"])
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
