import requests, json, bs4
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://zenith-prod-alt.ted.com/api/search"

def fetch_page(page):
    payload = [
        {
            "indexName": "popular",
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

def get_slugs():
    first_page = fetch_page(0)
    slugs = first_page[1]
    last_page_index = first_page[2]

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_page, i) for i in range(1, last_page_index + 1)]
        page_results = {}
        for future in futures:
            page_num, page_slugs, _ = future.result()
            page_results[page_num] = page_slugs
    
    for i in range(1, last_page_index + 1):
        slugs.extend(page_results[i])

    with open("data/slugs.json", "w") as f:
        json.dump(slugs, f, indent=4)
    
    print("Fetched slugs")

def fetch_speech_data(slug, build_id, session):
    url = f"https://www.ted.com/_next/data/{build_id}/talks/{slug}.json"
    res = session.get(url).json()
    data = res.get("pageProps", {}).get("videoData", {})

    if not data:
        redirect = res.get("pageProps", {})["__N_REDIRECT"]
        if redirect:
            url = f"https://www.ted.com/_next/data/{build_id}/dubbing/{slug}.json"
            res = session.get(url).json()
            data = res.get("pageProps", {}).get("videoData", {})
        else:
            print(slug)
            return None

    removed = [
        "takeaways", "talkExtras", "relatedVideos", "__typename",
        "primaryImageSet", "customContentDetails", "featured",
        "partnerName", "topics"
    ]
    for r in removed:
        data.pop(r, None)

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
    get_slugs()
    get_speech_data()