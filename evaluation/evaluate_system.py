"""Evaluation script to evaluate the whole system"""
import argparse

import pandas as pd

from model_pipelines import load_models
from main import send_input_to_system

model_map = load_models()


def evaluate(input_file, output_file, reference_file):
    with open(input_file, "r") as f:
        user_inputs = f.read().splitlines()

    answers = []
    for user_input in user_inputs:
        answers.append(send_input_to_system(model_map, user_input))

    with open(output_file, "w") as fp:
        for answer in answers:
            fp.write("%s\n" % answer)

    # TODO: evaluate using output file and reference file
    # https://github.com/Maluuba/nlg-eval/tree/master OR
    # https://github.com/Aldenhovel/bleu-rouge-meteor-cider-spice-eval4imagecaption


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate System")
    parser.add_argument("--data_path", required=True, help="location to dataset")
    parser.add_argument("--output_path", required=True, help="path to write generated answers")
    parser.add_argument("--ref_path", required=True, help="path to reference answers")

    args = parser.parse_args()

    dataset = pd.read_csv(args.data_path, delimiter="\t")
    evaluate(args.data_path, args.output_path, args.ref_path)
