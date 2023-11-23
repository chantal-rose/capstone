from transformers import pipeline ,set_seed

import argparse
import re
import pandas as pd
import torch
from model_pipelines import load_models
from main import send_input_to_system
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
import csv

from tqdm import tqdm

torch.cuda.empty_cache()
device = "cpu"
limit = 25
pipe = pipeline("text2text-generation", model="allenai/unifiedqa-t5-large",device=device)

set_seed(42)


def evaluate(input_file, output_file):
    eval_df = pd.read_csv(input_file)
    eval_df = eval_df.sample(frac=1)
    
    eval_df = eval_df.rename(columns={"Unnamed: 0": "image_id"})

    images = []
    annotations = []
    captions = []
    generated_answers = {"id":[], "output":[]}

    ctr = 0
    for index, row in tqdm(eval_df.iterrows(), desc="Evaluating datapoints"):
        if ctr == limit:
            break

        tqdm.write("############\n")
        tqdm.write("QUESTION: "+ row["question"]+"\n")
        tqdm.write("CONTEXT: "+ row["context"] +"\n")
        images.append({"id": str(row["image_id"])})
        annotations.append({
            "image_id": str(row["image_id"]),
            "id": str(row["image_id"]),
            "caption": row["answers"]
        })
       
        try:
            text = "question: {} context: {} answer: ".format(row["question"],  row["context"])
            system_output = pipe(text, max_length=4000, num_return_sequences=1, do_sample=True)
            system_output = system_output[0]["generated_text"]
        except Exception as e:
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

    for metric, score in coco_eval.eval.items():
        tqdm.write(f"{metric}: {score:.3f}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate System")
    parser.add_argument("--data_path", required=True, help="location to dataset")
    parser.add_argument("--output_path", required=True, help="path to write generated answers")

    args = parser.parse_args()

    dataset = pd.read_csv(args.data_path, delimiter="\t")
    evaluate(args.data_path, args.output_path)
