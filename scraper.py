import requests, json, bs4, sys, math
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://zenith-prod-alt.ted.com/api/search"

def fetch_page(page, sorting):
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
                "query": ""
            }
        }
    ]
    res = requests.post(BASE_URL, json=payload).json()
    hits = res["results"][0]["hits"]
    return page, [hit["slug"] for hit in hits], res["results"][0]["nbPages"]

def get_slugs(n_speeches=100, sorting="popular"):
    first_page = fetch_page(0, sorting)
    slugs = first_page[1]
    last_page_index = int(first_page[2])
    if n_speeches < last_page_index * 24:
        last_page_index = math.ceil(n_speeches / 24)

    if last_page_index != 0:
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(fetch_page, i, sorting) for i in range(1, last_page_index + 1)]
            page_results = {}
            for future in futures:
                page_num, page_slugs, _ = future.result()
                page_results[page_num] = page_slugs
    
    for i in range(1, last_page_index + 1):
        slugs.extend(page_results[i])

    with open("data/slugs.json", "w") as f:
        json.dump(slugs[:n_speeches], f, indent=4)
    
    print("Fetched slugs")

def fetch_speech_data(slug, build_id, session):
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
    data["streamUrl"] = json.loads(res_data["playerData"])["resources"]["hls"]["stream"]
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

    transcript_data = res.get("pageProps", {}).get("transcriptData", {})
    data["transcriptData"] = transcript_data

    return data

def get_speech_data():
    url = "https://www.ted.com"
    with requests.Session() as session:
        res = session.get(url)
        soup = bs4.BeautifulSoup(res.text, "lxml")
        next_data_tag = soup.find("script", id="__NEXT_DATA__")
        data = json.loads(next_data_tag.string)
        build_id = data["buildId"]

        with open("data/slugs.json") as slug_file:
            slugs = json.load(slug_file)

        speeches = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(fetch_speech_data, slug, build_id, session) for slug in slugs]
            for future in futures:
                speech_data = future.result()
                if speech_data:
                    speeches.append(speech_data)

    with open("data/speeches.json", "w") as f:
        json.dump(speeches, f, indent=4)

    print("Fetched speech data")

if __name__=="__main__":
    n_speeches = sys.argv[1] # number of speeches
    sorting = sys.argv[2] # "popular" for popular and "newest" for newest
    get_slugs(int(n_speeches), sorting)
    get_speech_data()