import myprosody as mysp
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Change to own path
c = r"/home/levomaaa/kurssit/intro-to-datascience/project/github_library/myprosody-master/myprosody"

# AudioFiles-folder
audio_dir = os.path.join(c, "dataset", "audioFiles")

wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]

results = []

def analyze_file(wav):
    p = os.path.splitext(wav)[0]
    try:
        print(f"\n>>> Analysing: {wav}")
        df = mysp.mysptotal(p, c)
        if df is not None:
            df = df.T
            df["file"] = wav
            return df
    except Exception as e:
        print(f"Error: {wav}: {e} (file skipped)")
    return None

with ProcessPoolExecutor(max_workers=16) as executor:  # Change max_workers as you wish
    future_to_wav = {executor.submit(analyze_file, wav): wav for wav in wav_files}
    for future in as_completed(future_to_wav):
        df = future.result()
        if df is not None:
            results.append(df)

# At the end, all result to a CSV-file
if results:
    final_df = pd.concat(results, ignore_index=True)
    combined_csv = os.path.join(audio_dir, "audio_analysis_1-2.csv")
    final_df.to_csv(combined_csv, index=False)
    print(f"\n All results put together to a CSV-file named: {combined_csv}")
else:
    print("No results")
