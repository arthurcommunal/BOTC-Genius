"""Download every article of the BOTC wiki through the MediaWiki API.

We save the raw API answer (rendered HTML + metadata) so that the cleaning 
can be changed and re-run without touching the network again.
"""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from botc.corpus.wiki_client import WikiClient

DEFAULT_OUT = Path("data/raw/wiki")
WIKI_URL = "https://wiki.bloodontheclocktower.com/"
EXCLUDED_TITLES = {"Main Page"}  # navigation only, no rules


def slugify(title: str) -> str:
    """'Devil's Advocate' -> 'Devil_s_Advocate' (safe as a file name)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def page_url(title: str) -> str:
    return WIKI_URL + title.replace(" ", "_")


def list_articles(client: WikiClient) -> list[dict]:
    """Return title, pageid and current revision id of every article (namespace 0, no redirects)."""
    pages = []
    for batch in client.query_all(generator="allpages", gapnamespace=0, gapfilterredir="nonredirects",
                                  gaplimit="max", prop="info"):
        pages.extend(batch["pages"])
    articles = [{"title": p["title"], "pageid": p["pageid"], "revid": p["lastrevid"]}
                for p in pages if p["title"] not in EXCLUDED_TITLES]
    return sorted(articles, key=lambda a: a["title"])


def fetch_page(client: WikiClient, pageid: int) -> dict:
    """Get the rendered HTML of one page with `action=parse`."""
    data = client.get(action="parse", pageid=pageid, prop="text|categories|revid",
                      disableeditsection=1, disabletoc=1)
    parse = data["parse"]
    return {
        "title": parse["title"],
        "pageid": parse["pageid"],
        "revid": parse["revid"],
        "url": page_url(parse["title"]),
        "categories": [c["category"].replace("_", " ") for c in parse["categories"]],
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "html": parse["text"],
    }


def is_up_to_date(path: Path, revid: int) -> bool:
    if not path.exists():
        return False
    return json.loads(path.read_text("utf-8"))["revid"] == revid


def write_json(path: Path, data: dict) -> None:
    """Write to a temporary file then rename it: a Ctrl-C never leaves a half-written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


def scrape(client: WikiClient, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    articles = list_articles(client)
    print(f"{len(articles)} articles on the wiki")

    downloaded = 0
    for i, article in enumerate(articles, start=1):
        path = out_dir / f"{slugify(article['title'])}.json"
        if is_up_to_date(path, article["revid"]):
            continue
        print(f"[{i}/{len(articles)}] {article['title']}")
        write_json(path, fetch_page(client, article["pageid"]))
        downloaded += 1

    print(f"Done: {downloaded} downloaded, {len(articles) - downloaded} already up to date, in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for the raw JSON files")
    parser.add_argument("--min-interval", type=float, default=2.0, help="seconds between two requests")
    args = parser.parse_args()

    with WikiClient(min_interval=args.min_interval) as client:
        scrape(client, args.out)


if __name__ == "__main__":
    main()
