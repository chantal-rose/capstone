"""Module that contains all helper functions for the system.
"""

from functools import lru_cache

from datasets import load_dataset
import json
import os.path
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
                             column_tuples: tuple,
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
    if not isinstance(column_tuples, tuple):
        raise Exception("Column names need to a list of column names as strings.")
    try:
        dataset = load_dataset(dataset, *args, **kwargs)
    except Exception as e:
        raise Exception("Error while loading dataset {}".format(e))
    shuffled_dataset = dataset.shuffle(seed=seed)
    df = pd.DataFrame(shuffled_dataset[:num_samples])
    try:
        return df[list(column_tuples)]
    except KeyError as e:
        raise e


def get_string_to_encode(data : dict):  
    """Returns a string which is a concatenation of model description, sample questions, and sample contexts from
    the dataset the model was trained on.

    :param data: dictionary object of the model's json file
    :return: string that is a concatenation of model description, sample questions, and sample contexts
    :raises Exception
    """  
    context_string = ""
    question_string = ""
    
    qa_tuple = ('question', 'context')
    open_domain_tuple = ('question')
    
    try:
        for dataset in data['dataset']:
            if data['task'].lower()=="qa":
                df = sample_rows_from_dataset(dataset, qa_tuple, split='validation')
                context_string = context_string + ' '.join(df['context'].tolist())
                question_string = question_string + ' '.join(df['question'].tolist())
            
            elif data['type'].lower()=="open domain":
                df = sample_rows_from_dataset(dataset, open_domain_tuple, split='validation')
                context_string = context_string + ' '.join(df['context'].tolist())
                question_string = question_string + ' '.join(df['question'].tolist())
    except:
        return ""
            
    return data['description'] + context_string + question_string
    

def create_map(force: bool = False, 
               new_files: list = []):
    """Creates a list of dictionary objects from the model's .json files. in addition to metadata from the .json file,
    it also populates the embedding of the model. 

    :param force: control flag to rerun the create map code for all .json files
    :new_files: list of files for which the create map function should run of force flag is false
    :return: persists the map to a file "model_map.json"
    """  
    
    model_directory = os.path.dirname(__file__) + "/models"
    
    try:
        with open(os.path.dirname(__file__) + '/model_map.json', "r") as f:
            past_map = json.load(f)
    except:
        past_map = []
    
    
    if force is True:
        new_files = os.listdir(model_directory)
        past_map = []
            
    model_file_list = [filename for filename in os.listdir(model_directory) if filename.endswith('.json') and filename in new_files]     

    for model_file in model_file_list:
        with open(model_directory + "/" + model_file) as model_json:
            data = json.load(model_json)
            data['embeddings'] = get_embeddings(get_string_to_encode(data)).tolist()
            past_map.append(data)            
    
    model_map_list = json.dumps(past_map)
        
    with open(os.path.dirname(__file__) + "/model_map.json", "w+") as f:
        f.write(model_map_list)
    
    
    
def get_map():
    """Gets a list of dictionaries of the current persisted state of the model map
    
    :return: list of dictionaries where each entry is meta_data of each map
    """  
    
    if "model_map.json" not in os.listdir(os.path.dirname(__file__)):
        create_map(force=True)
    
    try:
        with open(os.path.dirname(__file__) + '/model_map.json', "r") as f:
            model_map = json.load(f)
    except:
        model_map = []
            
    return model_map


if __name__ == "__main__":
    get_map()