from answer_verification import verify_answer, verify_answer_gpt

import pandas as pd
from tqdm import tqdm
import numpy as np

dataset = pd.read_csv('eval_dataset.csv')
np.random.seed(42)
dataset['shuffled_answer'] = np.random.permutation(dataset['answer'])

original_verify_boolq = []
shuffled_verify_boolq = []
original_verify_gpt = []
shuffled_verify_gpt = []

for pt in tqdm(dataset.to_dict(orient='records'), total=len(dataset)):
    original_verify_boolq.append(verify_answer(pt['question'], pt['context'], pt['answer']))
    shuffled_verify_boolq.append(not verify_answer(pt['question'], pt['context'], pt['shuffled_answer']))
    original_verify_gpt.append(verify_answer_gpt(pt['question'], pt['context'], pt['answer']))
    shuffled_verify_gpt.append(not verify_answer_gpt(pt['question'], pt['context'], pt['shuffled_answer']))

print(sum(x for x in original_verify_boolq))
print(sum(x for x in shuffled_verify_boolq))
print(sum(x for x in original_verify_gpt))
print(sum(x for x in shuffled_verify_gpt))

dataset['original_verify_boolq'] = original_verify_boolq
dataset['shuffled_verify_boolq'] = shuffled_verify_boolq
dataset['original_verify_gpt'] = original_verify_gpt
dataset['shuffled_verify_gpt'] = shuffled_verify_gpt

dataset.drop(columns=["context"]).to_csv('ans_ver.csv')
