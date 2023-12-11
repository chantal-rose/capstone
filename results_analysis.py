import pandas as pd
import sys
import argparse

def main(results):
    df_true = pd.read_csv('eval_dataset_fixed.csv', index_col='id')
    df = pd.read_csv(results, index_col='id').fillna("NOANSWER")

    full_df = df_true.join(df, how='right', validate="one_to_one")
    print(full_df['dataset'].value_counts())

    em = full_df["output"].str.lower() == full_df["answer"].str.lower()
    print("em:", sum(em))
    print(full_df[em]['dataset'].value_counts())

    subset_match = full_df.apply(lambda x: x['output'].lower() in x['answer'].lower() if len(x['output']) < len(x['answer']) else x['answer'].lower() in x['output'].lower(), axis=1)
    print("subset match:", sum(subset_match))
    print(full_df[subset_match]['dataset'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analyze Results")
    parser.add_argument("--results", required=True, help="results file 1")
    # parser.add_argument("--expected", required=True, help="expected answers")

    args = parser.parse_args()
    main(args.results)