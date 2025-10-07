import yt_dlp
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import re
import glob


class PlaylistScraper:
    def __init__(self, playlist_url):
        opts = {"quiet": True, "force_generic_extractor": True, "extract_flat": True}

        data = []
        with yt_dlp.YoutubeDL(opts) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
            vids = playlist_info.get("entries", [])

        for video in vids:
            if (
                video.get("duration") < 3601
            ):  # Ignore vids longer than 1 hour because myprosody freezes with long vids
                title = video.get("title")
                if title not in ("[Deleted video]", "[Private video]"):
                    data.append(
                        {
                            "title": re.sub(r'[<>:"/\\|?*]', "", title),
                            "url": video.get("url"),
                        }
                    )

        self.df = pd.DataFrame(data)
        with open("../data/playlist_data.json", "w") as f:
            json.dump(data, f, indent=4)

        self.last_idx = self.df.shape[0]
        self.curr_idx = 0

    def download_audio(self, title, url):
        out_path = os.path.join("../myprosody", "dataset", "audioFiles", title + ".mp4")
        wav_path = os.path.join("../myprosody", "dataset", "audioFiles", title + ".wav")

        yt_opts = {
            "format": "bestaudio",
            "outtmpl": out_path,
            "overwrites": True,
            "nopart": True,
        }

        try:
            with yt_dlp.YoutubeDL(yt_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"Skipping {title}: {e}")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    out_path,
                    "-ar",
                    "48000",
                    "-acodec",
                    "pcm_s32le",
                    wav_path,
                ],
                check=True,
            )
        except Exception as e:
            print(f"Skipping {title}: ffmpeg conversion failed: {e}")

        try:
            os.remove(out_path)
        except OSError:
            pass

    def get_audios(self, n):
        if self.curr_idx >= self.last_idx:
            return

        end_idx = min(self.curr_idx + n, self.last_idx)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.download_audio, vid["title"], vid["url"])
                for _, vid in self.df.iloc[self.curr_idx : end_idx].iterrows()
            ]
            for future in futures:
                future.result()

        self.curr_idx = end_idx

    def clear_audios(self):
        files = glob.glob("../myprosody/dataset/audioFiles/*")
        for f in files:
            os.remove(f)
