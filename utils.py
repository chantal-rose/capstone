"""Module that contains all helper functions for the system.
"""
from functools import lru_cache

from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


def compute_text_similarity(embedding1: torch.tensor, embedding2: torch.tensor) -> torch.Tensor:
    """Computes similarity between two pieces of text.

    :param embedding1: First encoded text (likely the query embedding)
    :param embedding2: Second encoded text
    :return: Similarity
    """
    return util.dot_score(embedding1, embedding2)


@lru_cache()
def fetch_embedding_model() -> SentenceTransformer:
    """Downloads the model for creating embeddings if not cached.

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
                             column_names: list,
                             num_samples: int = 100,
                             seed: int = 42,
                             *args, **kwargs) -> pd.DataFrame:
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
    if not isinstance(column_names, list):
        raise Exception("Column names need to a list of column names as strings.")
    try:
        dataset = load_dataset(dataset, *args, **kwargs)
    except Exception as e:
        raise Exception("Error while loading dataset {}".format(e))
    shuffled_dataset = dataset.shuffle(seed=seed)
    df = pd.DataFrame(shuffled_dataset[:num_samples])
    try:
        return df[column_names]
    except KeyError as e:
        raise e
