import requests
import json
import bs4
import math
import os
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor
import yt_dlp

BASE_URL = "https://zenith-prod-alt.ted.com/api/search"


class Scraper:
    def __init__(self, n_speeches=10, sorting="popular"):
        self.n_speeches = n_speeches
        self.current_speech_idx = 0
        self.get_slugs(int(n_speeches), sorting)
        self.get_speech_data()

    # this function is responsible for fetching the slugs (presenter + title) of a single search page
    def fetch_page(self, page, sorting):
        payload = [
            {
                "indexName": sorting,
                "params": {
                    "attributeForDistinct": "objectID",
                    "distinct": 1,
                    "facets": ["subtitle_languages", "tags"],
                    "highlightPostTag": "__/ais-highlight__",
                    "highlightPreTag": "__ais-highlight__",
                    "hitsPerPage": 24,
                    "maxValuesPerFacet": 500,
                    "page": page,
                    "query": "",
                },
            }
        ]
        res = requests.post(BASE_URL, json=payload).json()
        hits = res["results"][0]["hits"]
        return page, [hit["slug"] for hit in hits], res["results"][0]["nbPages"]

    # gets the slugs of the n first speeches sorted by the given criteria, and saves them to a json
    def get_slugs(self, n_speeches, sorting):
        first_page = self.fetch_page(0, sorting)
        slugs = first_page[1]
        last_page_index = int(first_page[2])
        if n_speeches < last_page_index * 24:
            last_page_index = math.ceil(n_speeches / 24)

        if last_page_index != 0:
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [
                    executor.submit(self.fetch_page, i, sorting)
                    for i in range(1, last_page_index + 1)
                ]
                page_results = {}
                for future in futures:
                    page_num, page_slugs, _ = future.result()
                    page_results[page_num] = page_slugs

        for i in range(1, last_page_index + 1):
            slugs.extend(page_results[i])

        with open("../data/slugs.json", "w") as f:
            json.dump(slugs[:n_speeches], f, indent=4)

        print("Fetched slugs")

    # fetches the speech data (like the audio stream url, title, views, etc.) of a single given speech
    def fetch_speech_data(self, slug, build_id, session):
        url = f"https://www.ted.com/_next/data/{build_id}/talks/{slug}.json"
        res = session.get(url).json()
        res_data = res.get("pageProps", {}).get("videoData", {})

        if not res_data:
            redirect = res.get("pageProps", {}).get("__N_REDIRECT", None)
            if redirect:
                url = f"https://www.ted.com/_next/data/{build_id}/dubbing/{slug}.json"
                res = session.get(url).json()
                res_data = res.get("pageProps", {}).get("videoData", {})
            else:
                print(slug)
                return None

        data = {}
        data["streamUrl"] = json.loads(res_data["playerData"])["resources"]["hls"][
            "stream"
        ]
        data["type_id"] = res_data["type"]["id"]
        data["type_name"] = res_data["type"][
            "name"
        ]  # probably unnecessary as the id essentially specifies the type
        data["socialTitle"] = res_data["socialTitle"]
        data["recordedOn"] = res_data["recordedOn"]
        data["language"] = res_data["language"]
        data["presenterDisplayName"] = res_data["presenterDisplayName"]
        data["duration"] = res_data["duration"]
        data["canonicalUrl"] = res_data["canonicalUrl"]
        data["viewedCount"] = res_data["viewedCount"]
        data["tedcomPercentage"] = res_data["tedcomPercentage"]
        data["youtubePercentage"] = res_data["youtubePercentage"]
        data["podcastsPercentage"] = res_data["podcastsPercentage"]
        data["tedappsPercentage"] = res_data["tedappsPercentage"]
        data["publishedAt"] = res_data["publishedAt"]
        data["id"] = res_data["id"]
        data["title"] = res_data["title"]
        data["slug"] = res_data["slug"]

        return data

    # gets the speech data of all of the associated slugs in the previously saved slugs.json
    def get_speech_data(self):
        url = "https://www.ted.com"
        with requests.Session() as session:
            res = session.get(url)
            soup = bs4.BeautifulSoup(res.text, "lxml")
            next_data_tag = soup.find("script", id="__NEXT_DATA__")
            data = json.loads(next_data_tag.string)
            build_id = data["buildId"]

            with open("../data/slugs.json") as slug_file:
                slugs = json.load(slug_file)

            speeches = []
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [
                    executor.submit(self.fetch_speech_data, slug, build_id, session)
                    for slug in slugs
                ]
                for future in futures:
                    speech_data = future.result()
                    if speech_data:
                        speeches.append(speech_data)

        with open("../data/speeches.json", "w") as f:
            json.dump(speeches, f, indent=4)

        print("Fetched speech data")

    # downloads and converts a single speech
    def download_audio(self, url, slug):
        out_path = os.path.join("../myprosody", "dataset", "audioFiles", slug + ".mp4")
        wav_path = os.path.join("../myprosody", "dataset", "audioFiles", slug + ".wav")

        yt_opts = {
            "format": "bestaudio",
            "outtmpl": out_path,
            "overwrites": True,
            "nopart": True,
        }

        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            ydl.download(url)

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

        os.remove(out_path)

    # gets the audios of all of the associated speeches in the previously saved speeches.json
    def get_audios(self, n_speeches):
        if not n_speeches:
            end_idx = self.n_speeches - 1
        elif self.current_speech_idx + n_speeches > self.n_speeches:
            end_idx = self.n_speeches - 1
        else:
            end_idx = self.current_speech_idx + n_speeches

        with open("../data/speeches.json", "r") as f:
            speech_data = json.load(f)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    self.download_audio, speech["streamUrl"], speech["slug"]
                )
                for speech in speech_data[self.current_speech_idx : end_idx]
                if "streamUrl" in speech and speech["streamUrl"]
            ]
            for future in futures:
                future.result()

        self.current_speech_idx += n_speeches

    # deletes the currently saved audios
    def clear_audios(self):
        files = glob.glob("../myprosody/dataset/audioFiles/*")
        for f in files:
            os.remove(f)
