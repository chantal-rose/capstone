"""Evaluation script to evaluate the whole system"""
import argparse

import pandas as pd
from transformers import set_seed

from llm_utils import domain_label
from main import send_input_to_system
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import json
import csv
import numpy as np

from tqdm import tqdm
import code_bert_score

import traceback

# model_map = load_models()
model_map = {}
limit = 5


def evaluate(input_file, output_file):
    eval_df = pd.read_csv(input_file)
    set_seed(42)
    eval_df = eval_df.sample(frac=1)
    
    eval_df = eval_df.rename(columns={"Unnamed: 0": "image_id"})

    images = []
    annotations = []
    captions = []

    predictions=[]
    ground_truth=[]
    generated_answers = {"id":[], "output":[]}

    ctr = 0
    for index, row in tqdm(eval_df.iterrows(), desc="Evaluating datapoints"):
        if ctr == limit:
            break

        tqdm.write("############\n")
        # tqdm.write("QUESTION: " + row["question"] + "\n")
        # tqdm.write("CONTEXT: " + row["context"] + "\n")
        images.append({"id": str(row["image_id"])})
        annotations.append({
            "image_id": str(row["image_id"]),
            "id": str(row["image_id"]),
            "caption": row["answer"]
        })
        if not isinstance(row["domain"], str):
            domain = "None"
        else:
            domain = row["domain"]
            #domain = domain_label(row["context"])
       
        try:
            system_output = send_input_to_system(model_map, row["question"], row["context"], domain)
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Exception in datapoint")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)
            traceback.print_exc()
            generated_answers["id"].append(row["image_id"])
            generated_answers["output"].append("ERROR")
            
            with open("results_generated.csv", "a", newline="") as csvfile:
                fieldnames = ["QID", "Question", "Domain", "Result"]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({"QID": row['question_id'],
                                 "Question": row["question"],
                                 "Domain": domain,
                                 "Result": "ERROR"
                                 })
            with open("results_expected.csv", "a", newline="") as csvfile:
                fieldnames = ["QID", "Question", "Domain", "Answer"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({"QID": row['question_id'],
                                 "Question": row["question"],
                                 "Domain": row['domain'],
                                 "Answer": row["answer"]})

            captions.append({
                "image_id": str(row["image_id"]),
                "caption": "ERROR"
            })
            predictions.append("ERROR")
            ground_truth.append(row["answer"])
            continue
        else:

            generated_answers["id"].append(row["image_id"])
            generated_answers["output"].append(system_output)
            
            with open("results_generated.csv", "a", newline="") as csvfile:
                fieldnames = ["QID", "Question", "Domain", "Result"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({"QID": row['question_id'],
                                 "Question": row["question"],
                                 "Domain": domain,
                                 "Result": system_output
                                 })
            with open("results_expected.csv", "a", newline="") as csvfile:
                fieldnames = ["QID", "Question", "Domain", "Answer"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({"QID": row['question_id'],
                                 "Question": row["question"],
                                 "Domain": row['domain'],
                                 "Answer": row["answer"]})

            captions.append({
                "image_id": str(row["image_id"]),
                "caption": system_output
            })
            predictions.append(system_output)
            ground_truth.append(row["answer"])
        ctr += 1
        
    references = {"images": images, "annotations": annotations}

    with open("references.json", "w") as f1:
        f1.write(json.dumps(references))

    with open("captions.json", "w") as f2:
        f2.write(json.dumps(captions))

    pd.DataFrame(generated_answers).to_csv(output_file, index=False)

    # create coco object and coco_result object
    annotation_file = "references.json"
    results_file = "captions.json"

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.evaluate()

    #evaluate on bert score
    pred_results = code_bert_score.score(cands=predictions, refs=ground_truth, lang='python')
    bert_score = {"precision":np.mean(pred_results[0].numpy()),"recall":np.mean(pred_results[1].numpy()),"f1":np.mean(pred_results[2].numpy()),"f3":np.mean(pred_results[3].numpy())}
    cocoEval_score = coco_eval.eval

    all_scores = {**bert_score,**cocoEval_score}
    for metric, score in all_scores.items():
        tqdm.write(f"{metric}: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate System")
    parser.add_argument("--data_path", required=True, help="location to dataset")
    parser.add_argument("--output_path", required=True, help="path to write generated answers")

    args = parser.parse_args()

    #dataset = pd.read_csv(args.data_path, delimiter=",")
    evaluate(args.data_path, args.output_path)
