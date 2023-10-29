from datasets import load_dataset

dataset = load_dataset("trivia_qa", "rc", split="validation")
print(dataset["entity_pages"][])
