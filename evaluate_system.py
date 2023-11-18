"""Evaluation script to evaluate the whole system"""
import argparse

import pandas as pd
from transformers import set_seed

from main import send_input_to_system
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
import csv

from tqdm import tqdm
import code_bert_score

# model_map = load_models()
model_map = {}
limit = 25


def evaluate(input_file, output_file):
    eval_df = pd.read_csv(input_file)
    set_seed(42)
    eval_df = eval_df.sample(frac=1)
    
    eval_df = eval_df.rename(columns={"Unnamed: 0": "image_id"})

    images = []
    annotations = []
    captions = []

    predictions=[]
    references=[]
    generated_answers = {"id":[], "output":[]}

    ctr = 0
    for index, row in tqdm(eval_df.iterrows(), desc="Evaluating datapoints"):
        if ctr == limit:
            break
        
        tqdm.write("############\n")
        tqdm.write("QUESTION: " + row["question"] + "\n")
        tqdm.write("CONTEXT: " + row["context"] + "\n")
        images.append({"id": str(row["image_id"])})
        annotations.append({
            "image_id": str(row["image_id"]),
            "id": str(row["image_id"]),
            "caption": row["answers"]
        })
        if not isinstance(row["domain"], str):
            domain = "None"
        else:
            domain = row["domain"]
       
        try:
            system_output = send_input_to_system(model_map, row["question"], row["context"], domain)
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Exception in datapoint")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)
            generated_answers["id"].append(row["image_id"])
            generated_answers["output"].append("ERROR")
            
            with open("results.csv", "a", newline="") as csvfile:
                fieldnames = ["Question", "Context", "Result", "Answer"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({"Question": row["question"],
                                 "Context": row["context"],
                                 "Result": "ERROR",
                                 "Answer": row["answers"]})

            captions.append({
                "image_id": str(row["image_id"]),
                "caption": "ERROR"
            })
            predictions.append("ERROR")
            references.append(row["answers"])
            continue
        else:

            generated_answers["id"].append(row["image_id"])
            generated_answers["output"].append(system_output)
            
            with open("results.csv", "a", newline="") as csvfile:
                fieldnames = ["Question", "Context", "Result", "Answer"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({"Question": row["question"],
                                 "Context": row["context"],
                                 "Result": system_output,
                                 "Answer": row["answers"]})

            captions.append({
                "image_id": str(row["image_id"]),
                "caption": system_output
            })
            predictions.append(system_output)
            references.append(row["answers"])
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
    pred_results = code_bert_score.score(cands=predictions, refs=references, lang='python')
    bert_score = {"precision":pred_results[0].numpy()[0],"recall":pred_results[1].numpy()[0],"f1":pred_results[0].numpy()[2],"f3":pred_results[3].numpy()[0]}
    cocoEval_score = coco_eval.eval

    all_scores = {**bert_score,**cocoEval_score}
    for metric, score in all_scores.items():
        tqdm.write(f"{metric}: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate System")
    parser.add_argument("--data_path", required=True, help="location to dataset")
    parser.add_argument("--output_path", required=True, help="path to write generated answers")

    args = parser.parse_args()

    dataset = pd.read_csv(args.data_path, delimiter="\t")
    evaluate(args.data_path, args.output_path)
