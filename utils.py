"""Module that contains all helper functions for the system.
"""
from functools import lru_cache
from heapq import nlargest
import json
from math import exp, pow, sqrt
import string
import os

from datasets import load_dataset
import pandas as pd
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import torch


nlp = spacy.load("en_core_web_lg")
stopwords = set(stopwords.words('english'))
punctuations = set(string.punctuation)


def get_cosine_similarity_score(answer1: str, answer2: str) -> float:
    """Returns cosine similarity between two answers.

    :param answer1: First answer to compare
    :param answer2: Second answer to compare
    :return: Cosine Similarity score
    """
    corpus = [answer1, answer2]
    vectorizer = TfidfVectorizer()
    sparse_matrix = vectorizer.fit_transform(corpus)

    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(
        doc_term_matrix,
        columns=vectorizer.get_feature_names_out(),
        index=["answer1", "answer2"],
    )
    return cosine_similarity(df, df)[0, 1]


def get_jaccard_index(answer1: str, answer2: str) -> float:
    """Returns jaccard similarity coefficent between two answers.

    :param answer1: First answer to compare
    :param answer2: Second answer to compare
    :return: Jaccard Similarity Coefficent
    """
    answer1 = set(nltk.word_tokenize(answer1))
    answer2 = set(nltk.word_tokenize(answer2))
    if not answer1 or not answer2:
        return 0
    return len(answer1 & answer2) / len(answer1 | answer2)


def get_euclidean_distance(answer1: str, answer2: str) -> float:
    """Returns Euclidean distance, or L2 norm between the embedding vectors of two answers.

    To compute the Euclidean distance we need vectors.
    So we use spaCy’s in-built Word2Vec model to create text embeddings.

    :param answer1: First answer to compare
    :param answer2: Second answer to compare
    :return: Euclidean distance
    """
    embeddings = [nlp(answer1).vector, nlp(answer2).vector]
    distance = sqrt(sum(pow(a - b, 2) for a, b in zip(embeddings[0], embeddings[1])))
    return 1 / exp(distance)


def get_answer_similarity_score(answer1: str, answer2: str, similarity_metric: str = "cosine_similarity") -> float:
    """Returns a similarity score between two answers.

    It can use different metrics to calculate the score.

    :param answer1: First answer to compare
    :param answer2: Second answer to compare
    :param similarity_metric: Metric to use for comparison
    :return: Similarity score
    """
    return SIMILARITY_METRIC_FUNCTION_MAP[similarity_metric](answer1, answer2)


def compute_similarity_between_embeddings(embedding1: torch.tensor, embedding2: torch.tensor) -> torch.Tensor:
    """Computes similarity between the embeddings of two pieces of text.

    :param embedding1: First encoded text (likely the query embedding)
    :param embedding2: Second encoded text
    :return: Similarity
    """
    return util.dot_score(embedding1, embedding2)


@lru_cache()
def fetch_embedding_model() -> SentenceTransformer:
    """Downloads the model for creating embeddings if not cached.

    Reference: https://www.sbert.net/docs/pretrained_models.html

    :return: Sentence Transformer model
    """
    return SentenceTransformer('multi-qa-MiniLM-L6-dot-v1')


def get_embeddings(text: str) -> np.ndarray:
    """Creates embeddings for text.

    :param text: Text to encode
    :return: Encoded Text as numpy array
    """
    return fetch_embedding_model().encode(text)


@lru_cache()
def sample_rows_from_dataset(dataset: str,
                             column_names: tuple,
                             *args,
                             num_samples: int = 100,
                             seed: int = 42,
                             **kwargs) -> pd.DataFrame:
    """Returns a dataframe of randomly sampled examples from a given dataset.

    :param dataset: HuggingFace dataset to download
    :param column_names: Columns to return (in most cases, it will be ['question', 'context'], but for open-domain,
        it will be ['question']
    :param num_samples: Number of rows to sample
    :param seed: Seed to use while shuffling dataset for reproducibility
    :param args: Arguments to pass to load_dataset function
    :param kwargs: Keyword arguments to pass to load_dataset function (like split="Train")
    :return: Pandas dataframe of randomly sampled examples
    :raises Exception
    """
    if not isinstance(column_names, tuple):
        raise Exception("Column names need to be a list of column names as strings.")
    try:
        dataset = load_dataset(dataset, *args, **kwargs)
    except Exception as e:
        print("Could NOT load dataset for {0}".format(dataset))
        raise Exception("Error while loading dataset {}".format(e))
    shuffled_dataset = dataset.shuffle(seed=seed)
    df = pd.DataFrame(shuffled_dataset[:num_samples])
    try:
        return df[list(column_names)]
    except KeyError as e:
        raise e


def get_string_from_row(row: pd.Series, columns: list) -> str:
    """Returns a string made from the column names and column values of each row

    :param row: Row from a dataframe
    :param columns: Tuple of column names
    :return: Combined string
    """
    s = ""
    for i, col in enumerate(columns):
        s += f"{col}: {str(row.iloc[i])} "

    return s


def get_string_to_encode(data: dict) -> str:
    """Returns a string which is a concatenation of model description, sample questions, and sample contexts from
    the dataset the model was trained on.

    :param data: Dictionary object of the model's json file
    :return: string that is a concatenation of model description, sample questions, and sample contexts
    :raises Exception
    """
    shuffled_string = ""
    for ind, dataset in enumerate(data["dataset"]):
        column_tuple = tuple(data["columns"][ind])
        config = None
        if "configs" in data:
            if data["configs"][ind] != "":
                config = data["configs"][ind]

        if config is not None:
            df = sample_rows_from_dataset(dataset, column_tuple, config, split=data['split'][ind])
        else:
            df = sample_rows_from_dataset(dataset, column_tuple, split=data['split'][ind])

        df = df.apply(get_string_from_row, args=(df.columns,), axis=1)
        shuffled_string += df.str.cat(sep=" ")
        shuffled_string = shuffled_string.replace("\n", "")
    
    total_string = data["description"] + shuffled_string
    word_tokens = word_tokenize(total_string)
    filtered_tokens = [w for w in word_tokens if not w.lower() in stopwords and not w.lower() in punctuations]
    return " ".join(filtered_tokens)


def create_map(filenames: list = None, force_recreate: bool = False) -> list:
    """Creates a list of dictionary objects from the model's .json files.

    In addition to metadata from the .json file, it also populates the embedding of the model.

    :param filenames: List of filenames provided if you want to limit the models for which you create a map
    :param force_recreate: True/False if you want to recreate the map if exists
    :return: JSON string of map of models as well as persists the map to a file "model_map.json"
    """
    repository_directory = os.path.dirname(__file__) + "/repository"
    past_map = []

    if "model_map.json" in os.listdir(os.path.dirname(__file__)):
        with open(os.path.dirname(__file__) + "/model_map.json", "r") as f:
            if not filenames:
                if not force_recreate:
                    return json.load(f)
                else:
                    filenames = os.listdir(repository_directory)
            elif filenames and not force_recreate:
                past_map = json.load(f)

    if not filenames or (filenames and force_recreate):
        filenames = os.listdir(repository_directory)

    model_file_list = [filename for filename in filenames if filename.endswith(".json")
                       and filename != "unifiedqaT5.json"]
    
    for model_file in model_file_list:
        with open(repository_directory + "/" + model_file) as model_json:
            data = json.load(model_json)
            data['embeddings'] = get_embeddings(get_string_to_encode(data)).tolist()
            past_map.append(data)            

    model_map_list = json.dumps(past_map)

    with open(os.path.dirname(__file__) + "/model_map.json", "w+") as f:
        f.write(model_map_list)

    return past_map


def get_top_k_models(question: str, context: str, k: int = 2) -> list:
    """Gets top k models based on similarity of embeddings to the embedding of the question+context

    :param question: Question from user
    :param context: Context from user
    :param k: Number of models required
    :return List of models
    """
    model_map = create_map()
    embedding = get_embeddings(f"question: {question} context: {context}")

    for model in model_map:
        model["similarity"] = compute_similarity_between_embeddings(embedding, model["embeddings"])

    def sort_key(x):
        return x["similarity"]

    best_models = nlargest(k, model_map, key=sort_key)
    return best_models


def filter_map(filter_field: str,
               field_val: str,
               k: int):
    """Returns a filtered map of top k models based on field (type/domain)

    :param filter_field: filter based on "type" or "domain"
    :param field_val: value to be filtered on
    :param k: top number of models to be returned
    :return: list of model dictionaries
    :raises Exception
    """
    model_map = create_map()
    filtered_models = []
    for model in model_map:

        if field_val in model[filter_field]:
            filtered_models.append(model)

    sorted_filtered_models = sorted(filtered_models, key=lambda x: x["downloads"], reverse=True)
    if len(sorted_filtered_models) < k:
        return sorted_filtered_models

    return sorted_filtered_models[:k]


def get_final_answer(answer_candidates: list,
                     confidence_score_of_candidates: list):
    """Returns a single answer from a list of candidates based on a custom formula based on confidence scores and 
    pairwise similarity.

    :param answer_candidates: List of answers from different models
    :param confidence_score_of_candidates: List of scores returned by the models
    :return: list of dictionaries where each entry is meta_data of each map
    """  
    max_score_idx = 0
    max_formula_score = -math.inf
    
    for i in range(len(answer_candidates)):
        curr_score = 0
        for j in range(i+1, len(answer_candidates)):
            curr_score += (confidence_score_of_candidates[j] * (get_answer_similarity_score(answer_candidates[i],
                                                                                            answer_candidates[j])))
        curr_score *= confidence_score_of_candidates[i]
        
        if curr_score > max_formula_score:
            max_formula_score = curr_score
            max_score_idx = i
            
    return answer_candidates[max_score_idx]


SIMILARITY_METRIC_FUNCTION_MAP = {
    "jaccard_index": get_jaccard_index,
    "cosine_similarity": get_cosine_similarity_score,
    "euclidean_distance": get_euclidean_distance,
}

if __name__ == "__main__":
    create_map()
