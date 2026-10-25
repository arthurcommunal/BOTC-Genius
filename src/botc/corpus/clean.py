"""Turn the raw pages downloaded by scrape.py into clean Markdown files."""

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

DEFAULT_RAW = Path("data/raw/wiki")
DEFAULT_OUT = Path("data/processed/wiki")

# Parts of the page that are not rules text.
REMOVED_SELECTORS = ["#character-details", "#edition-details", ".toc", ".html5audio", "figure", "img",
                     "script", "style", "input", "label"]
# Each of these ends with a blank line: one paragraph or one bullet = one block of text.
BLOCK_TAGS = ["p", "li", "dd", "div", "ul", "ol", "table", "tr", "dl"]


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for selector in REMOVED_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    for box in soup.select("#jinxes"):
        bullets = []
        for row in box.select("tr"):
            name, rule = [p.get_text(" ", strip=True) for p in row.select("p")]
            bullets.append(f"- {name}: {rule}")
        box.replace_with("\n\n## Jinxes\n\n" + "\n\n".join(bullets) + "\n\n")

    for heading in soup.find_all(["h2", "h3", "h4"]):
        level = int(heading.name[1])
        heading.replace_with(f"\n\n{'#' * level} {heading.get_text(' ', strip=True)}\n\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for item in soup.find_all(["li", "dd"]):
        item.insert(0, "- ")
    for cell in soup.find_all(["td", "th"]):
        cell.append(" ")
    for block in soup.find_all(BLOCK_TAGS):
        block.append("\n\n")

    return normalize_whitespace(soup.get_text())


def normalize_whitespace(text: str) -> str:
    """Collapse spaces inside each line and keep at most one blank line between paragraphs."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_page(page: dict) -> str:
    return f"# {page['title']}\n\n{html_to_markdown(page['html'])}\n"


def clean(raw_dir: Path, out_dir: Path) -> None:
    raw_paths = sorted(raw_dir.glob("*.json"))
    if not raw_paths:
        raise FileNotFoundError(f"No raw pages in {raw_dir}.")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "pages.jsonl", "w", encoding="utf-8") as manifest:
        for raw_path in raw_paths:
            page = json.loads(raw_path.read_text("utf-8"))
            md_path = out_dir / f"{raw_path.stem}.md"
            md_path.write_text(clean_page(page), "utf-8")
            meta = {key: page[key] for key in ("title", "pageid", "revid", "url", "categories")}
            meta["file"] = md_path.name
            manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"{len(raw_paths)} pages written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="directory written by scrape.py")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for the Markdown files")
    args = parser.parse_args()
    clean(args.raw, args.out)


if __name__ == "__main__":
    main()
