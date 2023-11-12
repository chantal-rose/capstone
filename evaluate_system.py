"""Evaluation script to evaluate the whole system"""
import argparse

import pandas as pd

from model_pipelines import load_models
from main import send_input_to_system
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json

# model_map = load_models()
model_map = {}

def evaluate(input_file, output_file):
    #input_file: csv with question,context,answer
    eval_df = pd.read_csv(input_file).tail(1)
    eval_df = eval_df.rename(columns={"Unnamed: 0": "image_id"})

    images = []
    annotations = []
    captions = []
    generated_answers = {'id':[], 'output':[]}

    ctr = 0
    for index, row in eval_df.iterrows():
        if(ctr == 5):
            break
        print("############")
        print("QUESTION: ", row['question'])
        print("CONTEXT: ", row['context'])
        images.append({"id": str(row['image_id'])})
        annotations.append({
            "image_id": str(row['image_id']),
            "id": str(row['image_id']),
            "caption": row['answers']
        })
        if not isinstance(row["domain"], str):
            domain = "None"
        else:
            domain = row["domain"]
        system_output = send_input_to_system(model_map, row['question'], row['context'], domain)

        generated_answers['id'].append(row['image_id'])
        generated_answers['output'].append(system_output)

        captions.append({
            "image_id": str(row['image_id']),
            "caption": system_output
        })
        ctr+=1
        
    references = {"images": images, "annotations": annotations}

    with open("references.json", "w") as f1:
        f1.write(json.dumps(references))

    with open("captions.json", "w") as f2:
        f2.write(json.dumps(captions))

    pd.DataFrame(generated_answers).to_csv(output_file, index=False)

    # create coco object and coco_result object
    annotation_file = 'references.json'
    results_file = 'captions.json'

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    # TODO: evaluate using output file and reference file
    # https://github.com/Maluuba/nlg-eval/tree/master OR
    # https://github.com/Aldenhovel/bleu-rouge-meteor-cider-spice-eval4imagecaption


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate System")
    parser.add_argument("--data_path", required=True, help="location to dataset")
    parser.add_argument("--output_path", required=True, help="path to write generated answers")


    args = parser.parse_args()

    dataset = pd.read_csv(args.data_path, delimiter="\t")
    evaluate(args.data_path, args.output_path)
