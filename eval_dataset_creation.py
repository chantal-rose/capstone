# %%
from datasets import load_dataset
import pandas as pd
import os
import json

# %%
repository_directory = os.path.abspath('') + "/repository"
models_jsons = os.listdir(repository_directory)

# %%
dataset_model_dict = {}

for model_file in models_jsons:
    with open(repository_directory + "/" + model_file) as model_json:
        data = json.load(model_json)
        for dataset in data['dataset']:
            if dataset not in dataset_model_dict:
                dataset_model_dict[dataset] = []
            
            dataset_model_dict[dataset].append(data['model_name'])

# %%
dataset_model_dict

# %%
def sample_rows_from_dataset(dataset: str,
                             column_names: tuple,
                             *args,
                             num_samples: int = 250,
                             seed: int = 42,
                             **kwargs) -> pd.DataFrame:    
    if not isinstance(column_names, tuple):
        raise Exception("Column names need to be a list of column names as strings.")
    try:
        dataset = load_dataset(dataset, *args, split="test")
    except Exception as e:
        print("Could NOT load dataset for {0}".format(dataset))
        raise Exception("Error while loading dataset {}".format(e))
    shuffled_dataset = dataset.shuffle(seed=seed)
    df = pd.DataFrame(shuffled_dataset[:num_samples])
    try:
        return df[list(column_names)]
    except KeyError as e:
        raise e
    
    
def sample_rows_from_dataset(dataset: str,
                             column_names: tuple,
                             *args,
                             num_samples: int = 3000,
                             seed: int = 42,
                             **kwargs) -> pd.DataFrame:
    if not isinstance(column_names, tuple):
        raise Exception("Column names need to be a list of column names as strings.")
    try:
        dataset = load_dataset(dataset, *args, **kwargs)
    except Exception as e:
        print("Could NOT load dataset for {0}".format(dataset))
        raise Exception("Error while loading dataset {}".format(e))
    shuffled_dataset = dataset.shuffle(seed=seed)
    df = pd.DataFrame(shuffled_dataset[:num_samples])
    try:
        return df[list(column_names)]
    except KeyError as e:
        raise e

# %% [markdown]
# ### Squad Dataset

# %%
dataset_name = "squad"
configs = None
column_tuple = ("question", "context", "answers")

squad_qa_dataset = sample_rows_from_dataset(dataset_name, column_tuple, split="validation")

# %%
answers = []

for i in range(len(squad_qa_dataset)):
    curr_ans_list = squad_qa_dataset['answers'][i]['text']
    curr_ans = max(curr_ans_list, key = len)
    answers.append(curr_ans)
    
squad_qa_dataset['answers'] = answers

# %% [markdown]
# ### Pubmed Biology Dataset

# %%
dataset_name = "pubmed_qa"
config = "pqa_labeled"
column_tuple = ("question", "context", "long_answer")

pubmed_qa_dataset = sample_rows_from_dataset(dataset_name, column_tuple, config, split="train")

contexts_strings = []

for i in range(len(pubmed_qa_dataset)):
    contexts_strings.append(' '.join(pubmed_qa_dataset["context"][i]['contexts']))
    
pubmed_qa_dataset['context'] = contexts_strings
pubmed_qa_dataset = pubmed_qa_dataset.rename(columns={"long_answer": "answers"})

# %% [markdown]
# ### BioASQ dataset

# %%
dataset_name = "BeIR/bioasq-generated-queries"
column_tuple = ("text", "query")

bioasq_qa_dataset = sample_rows_from_dataset(dataset_name, column_tuple, split="train")
bioasq_qa_dataset = bioasq_qa_dataset.rename(columns={"text": "context", "query": "question"})
bioasq_qa_dataset = bioasq_qa_dataset[["question", "context"]]

# %% [markdown]
# ### cuad (legal) dataset

# %%
dataset_name = "cuad"
column_tuple = ("question", "context", "answers")

cuad_qa_dataset = sample_rows_from_dataset(dataset_name, column_tuple, split="train")

# %%
answers = []

for i in range(len(cuad_qa_dataset)):
    curr_ans_list = cuad_qa_dataset['answers'][i]['text']
    if len(curr_ans_list)!=0:
        curr_ans = max(curr_ans_list, key = len)
    else:
        curr_ans = ""
    answers.append(curr_ans)
    
cuad_qa_dataset['answers'] = answers

# %%
cuad_qa_dataset = cuad_qa_dataset[cuad_qa_dataset["answers"]!=""][:250]

# %% [markdown]
# ### Combining datasets

# %%
cuad_qa_dataset["domain"] = "legal"
#bioasq_qa_dataset["domain"] = "bio"
pubmed_qa_dataset["domain"] = "bio"
squad_qa_dataset["domain"] = "None"

# %%
dataset_model_dict['pubmed_qa']

# %%
cuad_qa_dataset['models'] = cuad_qa_dataset['models'].apply(lambda x: dataset_model_dict['cuad'])
pubmed_qa_dataset['models'] = pubmed_qa_dataset['models'].apply(lambda x: dataset_model_dict['pubmed_qa'])
squad_qa_dataset['models'] = squad_qa_dataset['models'].apply(lambda x: dataset_model_dict['squad'])


# %%
eval_dataset = pd.concat([cuad_qa_dataset, pubmed_qa_dataset, squad_qa_dataset], ignore_index=True)

# %%
eval_dataset.tail(10)

# %%
eval_dataset.to_csv("eval_dataset.csv")


