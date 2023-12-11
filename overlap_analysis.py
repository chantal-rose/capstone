import pandas as pd
import sys
import argparse

def main(results1, results2):
    df_true = pd.read_csv('eval_dataset_fixed.csv', index_col='id')
    df1 = pd.read_csv(results1, index_col='id').rename(columns={"output": "output1"}).fillna("NOANSWER")
    df2 = pd.read_csv(results2, index_col='id').rename(columns={"output": "output2"}).fillna("NOANSWER")

    full_df = df_true.join([df1, df2], how='left', validate="one_to_one")
    em1 = full_df["output1"].str.lower() == full_df["answer"].str.lower()
    em2 = full_df["output2"].str.lower() == full_df["answer"].str.lower()
    print("em1:", sum(em1))
    print("em2:", sum(em2))
    overlap = em1 & em2
    print("overlap:", sum(overlap))
    em1_only = em1 & ~em2
    print("em1 only:", sum(em1_only))
    em2_only = ~em1 & em2
    print("em2 only:", sum(em2_only))
    subset_match1 = full_df.apply(lambda x: x['output1'].lower() in x['answer'].lower() if len(x['output1']) < len(x['answer']) else x['answer'].lower() in x['output1'].lower(), axis=1)
    print("subset match1:", sum(subset_match1))
    subset_match2 = full_df.apply(lambda x: x['output2'].lower() in x['answer'].lower() if len(x['output2']) < len(x['answer']) else x['answer'].lower() in x['output2'].lower(), axis=1)
    print("subset match2:", sum(subset_match2))
    overlap = subset_match1 & subset_match2
    print("overlap:", sum(overlap))
    subset_match1_only = subset_match1 & ~subset_match2
    print("subset_match1 only:", sum(subset_match1_only))
    subset_match2_only = ~subset_match1 & subset_match2
    print("subset_match2 only:", sum(subset_match2_only))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analyze Results")
    parser.add_argument("--results1", required=True, help="results file 1")
    parser.add_argument("--results2", required=True, help="results file 2")
    # parser.add_argument("--expected", required=True, help="expected answers")

    args = parser.parse_args()

    main(args.results1, args.results2)#, args.expected)