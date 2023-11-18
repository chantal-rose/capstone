from answer_verification import verify_answer

import pandas as pd
from tqdm import tqdm
import numpy as np

dataset = pd.read_csv('eval_dataset_v2.csv')
np.random.seed(42)
dataset['shuffled_answers'] = np.random.permutation(dataset['answers'])

original_verify = []
shuffled_verify = []

for pt in tqdm(dataset.to_dict(orient='records')):
    if type(pt['context']) == float:
        original_verify.append("NA")
        shuffled_verify.append("NA")
        continue
    if verify_answer(pt['question'], pt['context'], pt['answers']):
        original_verify.append("T")
    else:
        original_verify.append("F")
    if not verify_answer(pt['question'], pt['context'], pt['shuffled_answers']):
        shuffled_verify.append("T")
    else:
        shuffled_verify.append("F")

print(sum(x=="T" for x in original_verify))
print(sum(x=="T" for x in shuffled_verify))

if len(original_verify) != len(dataset) or len(shuffled_verify) != len(dataset):
    with open("ans_ver_original.txt", 'w') as f:
        for r in original_verify:
            f.write(f"{r}\n")
    with open("ans_ver_shuffled.txt", 'w') as f:
        for r in shuffled_verify:
            f.write(f"{r}\n")
else:
    dataset['original_verify'] = original_verify
    dataset['shuffled_verify'] = shuffled_verify
    dataset.to_csv('ans_ver.csv')
