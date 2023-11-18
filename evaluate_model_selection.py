"""Evaluation script to evaluate model selection based on similarity of data point to model embeddings."""
import argparse
from heapq import nlargest

import pandas as pd

from utils import compute_similarity_between_embeddings
from utils import create_map
from utils import get_embeddings
import os

import sys 


# Python program to illustrate the intersection
# of two lists in most simple way
def intersection(lst1, lst2):
    intersection_list = [value for value in lst1 if value in lst2]
    return intersection_list


def evaluate(df, k):
    correct = 0
    total = len(dataset)
    model_map = create_map()

    for i, row in df.iterrows():
        question = row["question"]
        context = row["context"]
        embedding = get_embeddings(f"question: {question} context: {context}")
        for model in model_map:
            model["similarity"] = compute_similarity_between_embeddings(embedding, model["embeddings"]).item()

        best_models = nlargest(k, model_map, key=lambda x: x["similarity"])
        model_names = [model["model_name"] for model in best_models]
                
        if len(intersection(model_names, row["models"])):
            correct += 1
            
        if i % 50 == 0:
            print(i)

    print("Accuracy: ", correct / total)


if __name__ == "__main__":
    sys.path.append(os.path.abspath(''))
    parser = argparse.ArgumentParser("Evaluate Model Selection")
    parser.add_argument("--data_path", required=True, help="location to dataset")
    parser.add_argument("--n", required=True, help="number of models to choose")

    args = parser.parse_args()

    dataset = pd.read_csv(args.data_path, delimiter=",")
    n = int(args.n)
    evaluate(dataset, n)
