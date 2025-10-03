import sys
import glob
import os
import pandas as pd
import json
from playlist_analyze import analyze


# merges the csv files created by the analyze function, and joins the playlist data collected into playlist_data.json, saves everything into analyzed_playlist.csv
def merge_and_join():
    csv_files = glob.glob(os.path.join("../data/csv", "*.csv"))
    if not csv_files:
        print("No CSVs to merge")
        return
    try:
        dfs = []
        for f in csv_files:
            if os.path.basename(f) != "analyzed_speeches.csv":
                df = pd.read_csv(f, index_col=False)
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                dfs.append(df)
        merged_df = pd.concat(dfs, ignore_index=True)
        with open("../data/playlist_data.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)
        json_df = pd.DataFrame(json_data)
        merged_df = pd.merge(merged_df, json_df, on="title", how="left")
        merged_csv_path = os.path.join("../data/csv", "analyzed_playlist.csv")
        merged_df.to_csv(merged_csv_path, index=False)
        print("Merged CSVs and wrote to ../data/csv/analyzed_playlist.csv")
        for f in csv_files:
            if os.path.basename(f) not in [
                "analyzed_speeches.csv",
                "analyzed_playlist.csv",
            ]:
                os.remove(f)
    except Exception as e:
        print("Error merging:", e)
        return


def main(url, n_per_time):
    analyze(url, n_per_time)
    merge_and_join()


# Usage: python playlist_scrape_and_analyze.py [url: str (playlist url to analyze)] [n_per_time: int (amount of vids to download and analyze at once)]
if __name__ == "__main__":
    n_per_time = 4
    if len(sys.argv) > 1:
        url = str(sys.argv[1])  # url of playlist
    if len(sys.argv) > 2:
        n_per_time = int(sys.argv[2])  # how many vids are processed at once
    else:
        raise ValueError("Provide the URL for the playlist to analyze!")
    main(url, n_per_time)
