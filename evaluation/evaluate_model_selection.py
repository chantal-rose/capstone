"""Evaluation script to evaluate model selection based on similarity of data point to model embeddings."""
import argparse
from heapq import nlargest

import pandas as pd

from utils import compute_similarity_between_embeddings
from utils import create_map
from utils import get_embeddings


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

        if row["ground_truth"] in model_names:  # may be more than one ground-truth
            correct += 1

    print("Accuracy: ", correct / total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate Model Selection")
    parser.add_argument("--data_path", required=True, help="location to dataset")
    parser.add_argument("--n", required=True, help="number of models to choose")

    args = parser.parse_args()

    dataset = pd.read_csv(args.data_path, delimiter="\t")
    n = int(args.n)
    evaluate(dataset, n)
