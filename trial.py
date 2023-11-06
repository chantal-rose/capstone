from datasets import load_dataset

dataset = load_dataset("trivia_qa", "rc", split="validation", streaming=True, download_mode="force_redownload")
ds = dataset.take(2)

for row in ds:
    print(row)
    break

