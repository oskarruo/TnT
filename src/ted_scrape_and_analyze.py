import sys
import glob
import os
import pandas as pd
import json
import re
from analyzers.ted_analyze import analyze


# merges the csv files created by the analyze function, and joins the speech data collected into speeches.json, saves everything into analysis.csv
def merge_and_join(n_speeches, sorting):
    csv_files = glob.glob(os.path.join("../data/csv", "*.csv"))
    if not csv_files:
        print("No CSVs to merge")
        return
    try:
        dfs = []
        for f in csv_files:
            if re.match(r"^analysis_", os.path.basename(f)):
                df = pd.read_csv(f, index_col=False)
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                dfs.append(df)
        merged_df = pd.concat(dfs, ignore_index=True)
        with open("../data/speeches.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)
        json_df = pd.DataFrame(json_data)
        merged_df = pd.merge(merged_df, json_df, on="slug", how="left")
        merged_csv_path = os.path.join(
            "../data/csv", f"analyzed_speeches_{n_speeches}_{sorting}.csv"
        )
        merged_df.to_csv(merged_csv_path, index=False)
        print(
            f"Merged CSVs and wrote to ../data/csv/analyzed_speeches__{n_speeches}_{sorting}.csv"
        )
        for f in csv_files:
            filename = os.path.basename(f)
            if not re.match(r"^analyzed_", filename):
                os.remove(f)
    except Exception as e:
        print("Error merging:", e)
        return


def main(n_speeches, n_per_time, sorting):
    analyze(n_speeches, n_per_time, sorting)
    merge_and_join(n_speeches, sorting)


# Usage: python ted_scrape_and_analyze.py [n: int (amount of speeches to download)] [n_per_time: int (amount of speeches to download and analyze at once)] [sorting :string (sort by "popular" or "newest" speeches)]
if __name__ == "__main__":
    n_speeches = 10
    n_per_time = 5
    sorting = "popular"
    if len(sys.argv) > 1:
        n_speeches = int(sys.argv[1])  # number of speeches
    if len(sys.argv) > 2:
        n_per_time = int(sys.argv[2])  # how many speeches are processed at once
    if len(sys.argv) > 3:
        sorting = sys.argv[3]  # "popular" for popular and "newest" for newest
    main(n_speeches, n_per_time, sorting)
