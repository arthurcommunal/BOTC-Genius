"""Cut the clean Markdown pages written by clean.py into chunks for retrieval.

Structural chunking: a chunk never crosses a heading, and a paragraph or a bullet is never cut
in two unless it is longer than `max_words` on its own (then it is cut between sentences).
Inside a section, consecutive blocks are grouped until the chunk would exceed `max_words`.

Each chunk gets a deterministic contextual header (page, character type, edition, section,
ability) that is prepended to the text before embedding.
"""

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROCESSED = Path("data/processed/wiki")
DEFAULT_MAX_WORDS = 200  # ~270 tokens: far below the 512-token limit of BERT-sized embedders

HEADING = re.compile(r"^(#{1,6}) (.+)$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Wiki categories -> metadata. The other categories are ignored.
CHARACTER_TYPES = {"Townsfolk": "Townsfolk", "Outsiders": "Outsider", "Minions": "Minion",
                   "Demons": "Demon", "Travellers": "Traveller", "Fabled": "Fabled", "Loric": "Loric"}
EDITIONS = ["Trouble Brewing", "Bad Moon Rising", "Sects & Violets", "Experimental Characters"]
EXCLUDED_PAGES = {"Changelog"}


@dataclass
class Chunk:
    id: str                     # "Imp#3": page file + position in the page
    title: str                  # "Imp"
    section: str                # "Tips & Tricks", "" before the first heading
    url: str                    # link to the section on the wiki
    character_type: str | None  # "Demon", None for rules pages
    editions: list[str]         # ["Trouble Brewing"]
    revid: int                  # wiki revision the text comes from
    header: str                 # contextual header, see make_header()
    text: str                   # the chunk itself, as written on the wiki

    def embedding_text(self) -> str:
        return f"{self.header}\n\n{self.text}"


def count_words(text: str) -> int:
    return len(text.split())


def split_sections(markdown: str) -> list[tuple[str, list[str]]]:
    """Return (section path, blocks) for each section, in page order.

    The path joins the nested headings: "Djinn Special Rules > Townsfolk".
    The page title ("# Title") is skipped: it is already in the metadata.
    """
    sections = []
    path: list[str] = []
    blocks: list[str] = []
    for block in markdown.split("\n\n"):
        block = block.strip()
        match = HEADING.match(block)
        if match is None:
            if block:
                blocks.append(block)
            continue
        # A heading closes the current section and opens a new one.
        if blocks:
            sections.append((" > ".join(path), blocks))
        blocks = []
        level = len(match.group(1))
        if level >= 2:  # "##" is the first level below the title
            path = path[:level - 2] + [match.group(2)]
    if blocks:
        sections.append((" > ".join(path), blocks))
    return sections


def pack(pieces: list[str], max_words: int, separator: str) -> list[str]:
    """Group consecutive pieces greedily, as long as a group stays within `max_words`.

    A piece is never cut: a single piece longer than `max_words` makes a group on its own.
    """
    groups = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        n = count_words(piece)
        if current and size + n > max_words:
            groups.append(separator.join(current))
            current, size = [], 0
        current.append(piece)
        size += n
    if current:
        groups.append(separator.join(current))
    return groups


def split_long_block(block: str, max_words: int) -> list[str]:
    """A block (paragraph or bullet) that is too long is cut between sentences."""
    if count_words(block) <= max_words:
        return [block]
    return pack(SENTENCE_END.split(block), max_words, " ")


def chunk_section(blocks: list[str], max_words: int) -> list[str]:
    pieces = [piece for block in blocks for piece in split_long_block(block, max_words)]
    return pack(pieces, max_words, "\n\n")


def find_ability(sections: list[tuple[str, list[str]]]) -> str | None:
    """On a character page, the first block of "Summary" is the ability text, in quotes."""
    for path, blocks in sections:
        if path == "Summary" and blocks[0].startswith('"'):
            return blocks[0]
    return None


def make_header(title: str, character_type: str | None, editions: list[str], section: str,
                ability: str | None, text: str) -> str:
    """Deterministic template, e.g.

        Imp (Demon, Trouble Brewing) > How to Run
        Ability: "Each night*, choose a player: they die. ..."
    """
    tags = ([character_type] if character_type else []) + editions
    line = f"{title} ({', '.join(tags)})" if tags else title
    if section:
        line += f" > {section}"
    if ability and ability not in text:  # the Summary chunk already contains it
        line += f"\nAbility: {ability}"
    return line


def chunk_page(page: dict, markdown: str, max_words: int = DEFAULT_MAX_WORDS) -> list[Chunk]:
    """`page` is one line of pages.jsonl (title, revid, url, categories, file)."""
    categories = page["categories"]
    character_type = next((CHARACTER_TYPES[c] for c in categories if c in CHARACTER_TYPES), None)
    editions = [e for e in EDITIONS if e in categories]
    sections = split_sections(markdown)
    ability = find_ability(sections)
    stem = Path(page["file"]).stem

    chunks = []
    for section, blocks in sections:
        anchor = section.split(" > ")[-1].replace(" ", "_")
        url = f"{page['url']}#{anchor}" if section else page["url"]
        for text in chunk_section(blocks, max_words):
            chunks.append(Chunk(
                id=f"{stem}#{len(chunks)}",
                title=page["title"],
                section=section,
                url=url,
                character_type=character_type,
                editions=editions,
                revid=page["revid"],
                header=make_header(page["title"], character_type, editions, section, ability, text),
                text=text,
            ))
    return chunks


def chunk_corpus(processed_dir: Path, max_words: int = DEFAULT_MAX_WORDS) -> list[Chunk]:
    """Chunk every page listed in pages.jsonl (written by clean.py)."""
    manifest = processed_dir / "pages.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found: run botc.corpus.clean first.")
    chunks = []
    for line in manifest.read_text("utf-8").splitlines():
        page = json.loads(line)
        if page["title"] in EXCLUDED_PAGES:
            continue
        markdown = (processed_dir / page["file"]).read_text("utf-8")
        chunks.extend(chunk_page(page, markdown, max_words))
    return chunks


def print_stats(chunks: list[Chunk]) -> None:
    sizes = [count_words(c.embedding_text()) for c in chunks]
    pages = {c.title for c in chunks}
    print(f"{len(chunks)} chunks from {len(pages)} pages")
    print(f"words per chunk (header included): min {min(sizes)}, median {statistics.median(sizes):.0f}, "
          f"max {max(sizes)}")
    print(f"chunks under 30 words: {sum(s < 30 for s in sizes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED, help="directory written by clean.py")
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--show", metavar="TITLE", help="print the chunks of this page")
    args = parser.parse_args()

    chunks = chunk_corpus(args.processed, args.max_words)
    if args.show is None:
        print_stats(chunks)
        return
    for chunk in chunks:
        if chunk.title == args.show:
            print(f"=== {chunk.id}  ({count_words(chunk.text)} words)  {chunk.url}")
            print(chunk.embedding_text(), end="\n\n")


if __name__ == "__main__":
    main()
