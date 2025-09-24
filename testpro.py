import myprosody as mysp
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

c = os.path.abspath("./myprosody")

# AudioFiles-folder
audio_dir = os.path.join(c, "dataset", "audioFiles")

# Get all .wav files
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

            # Split filename into title, author, and id_part
            parts = [part.strip() for part in p.split("｜")]
            df["title"] = parts[0] if len(parts) > 0 else ""
            df["author"] = parts[1] if len(parts) > 1 else ""
            df["id_part"] = parts[2] if len(parts) > 2 else ""

            return df
    except Exception as e:
        print(f"Error: {wav}: {e} (file skipped)")
    return None

# Process files in parallel
with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust max_workers as needed
    future_to_wav = {executor.submit(analyze_file, wav): wav for wav in wav_files}
    for future in as_completed(future_to_wav):
        df = future.result()
        if df is not None:
            results.append(df)

# Save combined results to a single CSV
if results:
    final_df = pd.concat(results, ignore_index=True)
    combined_csv = os.path.join(audio_dir, "audio_analysis_1-3.csv")
    final_df.to_csv(combined_csv, index=False)
    print(f"\nAll results put together to a CSV-file named: {combined_csv}")
else:
    print("No results")
