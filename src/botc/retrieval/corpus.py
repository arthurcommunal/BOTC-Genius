"""Process the corpus of documents (.md files)"""

from pathlib import Path


def load_corpus(directory: Path) -> list[tuple[str, str]]:
    corpus = []
    paths = sorted(directory.glob("*.md"))
    if not directory.is_dir() or len(paths) == 0:
        raise FileNotFoundError(f"There are no rules files in {directory}")
    for path in paths:
        text = path.read_text("utf-8")
        sections_raw = text.split("\n\n")
        sections_strip = [s.strip() for s in sections_raw]
        sections = [s for s in sections_strip if len(s) > 0]
        ids = [f"{path.stem}#{i}" for i in range(len(sections))]
        corpus.extend(zip(ids, sections))
    if len(corpus) == 0:
        raise ValueError("The corpus is empty")
    return corpus
