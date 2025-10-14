import myprosody as mysp
import os
import pandas as pd
import io
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from scrapers.playlist_scraper import PlaylistScraper


# parses the output string of mysptotal
def parse_mysptotal_output(output, wav):
    lines = [line.strip() for line in output.splitlines() if line.strip()]

    if any("Try again" in line for line in lines):
        print(f"Failed {wav}, skipping.")
        return None

    if len(lines) > 1:
        rows = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            metric = " ".join(parts[:-1])
            value = parts[-1]
            rows.append((metric, value))

        if rows:
            df = pd.DataFrame([dict(rows)])
            df["title"] = os.path.splitext(os.path.basename(wav))[0]
            return df

    return None


# analyzes a single file
def analyze_file(wav, c):
    p = os.path.splitext(wav)[0]
    try:
        print(f"\n>>> Analyzing: {wav}")

        # the mysp outputs are just strings that are printed so the stdout has to be captured
        buf = io.StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buf
        mysp.mysptotal(p, c)
        sys.stdout = sys_stdout
        output = buf.getvalue()
        buf.close()

        return parse_mysptotal_output(output, wav)

    except Exception as e:
        print(f"Error: {wav}: {e} (file skipped)")
    return None


# analyzes audios
def analyze(url, n_per_time=5):
    c = os.path.abspath("../myprosody")
    # AudioFiles-folder
    audio_dir = os.path.join(c, "dataset", "audioFiles")

    workers = n_per_time if n_per_time <= 20 else 20  # limit workers to 20 for now

    s = PlaylistScraper(url)
    n = s.last_idx

    # analyze n_per_time audios per time (because of space constraints)
    for start in range(0, n, n_per_time):
        end = min(start + n_per_time - 1, n - 1)
        s.get_audios(n_per_time)
        wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
        results = []

        # Process files in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_wav = {
                executor.submit(analyze_file, wav, c): wav for wav in wav_files
            }
            for future in as_completed(future_to_wav):
                df = future.result()
                if df is not None:
                    results.append(df)

        # Save combined results to a CSV
        if results:
            final_df = pd.concat(results, ignore_index=True)
            final_df.to_csv(f"../data/csv/analysis_{start}_{end}.csv", index=False)
            print(f"\nResults of range {start}-{end} put together to a CSV-file")
            print(f"Progress {(end + 1) / n * 100}%")
        else:
            print("No results")

        s.clear_audios()
