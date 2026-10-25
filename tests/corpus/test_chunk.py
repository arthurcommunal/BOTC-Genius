import json

from botc.corpus.chunk import chunk_corpus, chunk_page, count_words, pack, split_sections

PAGE = {"title": "Imp", "pageid": 1, "revid": 42, "url": "https://wiki.test/Imp",
        "categories": ["Demons", "Trouble Brewing"], "file": "Imp.md"}

MARKDOWN = """# Imp

## Summary

"Each night*, choose a player: they die."

- First fact.

## How to Run

Wake the Imp.

### Special case

Do something else."""


def test_sections_follow_the_heading_path():
    assert split_sections(MARKDOWN) == [
        ("Summary", ['"Each night*, choose a player: they die."', "- First fact."]),
        ("How to Run", ["Wake the Imp."]),
        ("How to Run > Special case", ["Do something else."]),
    ]


def test_text_before_the_first_heading_has_an_empty_path():
    assert split_sections("# Glossary\n\nAbility: a power.\n\nAlive: not dead.") == [
        ("", ["Ability: a power.", "Alive: not dead."]),
    ]


def test_pack_groups_blocks_without_cutting_them():
    assert pack(["a b", "c d", "e f"], max_words=4, separator="\n\n") == ["a b\n\nc d", "e f"]


def test_a_block_longer_than_the_limit_stays_alone():
    assert pack(["a b c d e", "f"], max_words=3, separator="\n\n") == ["a b c d e", "f"]


def test_a_long_paragraph_is_cut_between_sentences():
    long = " ".join(f"Sentence number {i} is here." for i in range(20))  # 100 words
    chunks = chunk_page(PAGE, f"# Imp\n\n## Summary\n\n{long}", max_words=30)
    assert len(chunks) == 4
    assert all(count_words(c.text) <= 30 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_a_chunk_never_crosses_a_heading():
    chunks = chunk_page(PAGE, MARKDOWN, max_words=1000)
    assert [c.section for c in chunks] == ["Summary", "How to Run", "How to Run > Special case"]


def test_metadata_and_ids():
    chunks = chunk_page(PAGE, MARKDOWN)
    assert [c.id for c in chunks] == ["Imp#0", "Imp#1", "Imp#2"]
    assert chunks[1].character_type == "Demon"
    assert chunks[1].editions == ["Trouble Brewing"]
    assert chunks[1].url == "https://wiki.test/Imp#How_to_Run"
    assert chunks[2].url == "https://wiki.test/Imp#Special_case"


def test_header_gives_page_type_edition_section_and_ability():
    chunks = chunk_page(PAGE, MARKDOWN)
    assert chunks[1].header == ('Imp (Demon, Trouble Brewing) > How to Run\n'
                                'Ability: "Each night*, choose a player: they die."')
    assert chunks[1].embedding_text() == chunks[1].header + "\n\nWake the Imp."


def test_ability_is_not_repeated_in_the_summary_header():
    assert chunk_page(PAGE, MARKDOWN)[0].header == "Imp (Demon, Trouble Brewing) > Summary"


def test_rules_page_header_has_no_tags():
    page = {**PAGE, "title": "Glossary", "categories": [], "file": "Glossary.md"}
    chunk = chunk_page(page, "# Glossary\n\nAlive: not dead.")[0]
    assert chunk.header == "Glossary"
    assert chunk.character_type is None


def test_chunk_corpus_reads_the_manifest_and_skips_the_changelog(tmp_path):
    changelog = {**PAGE, "title": "Changelog", "categories": [], "file": "Changelog.md"}
    lines = [json.dumps(PAGE), json.dumps(changelog)]
    (tmp_path / "pages.jsonl").write_text("\n".join(lines), "utf-8")
    (tmp_path / "Imp.md").write_text(MARKDOWN, "utf-8")
    (tmp_path / "Changelog.md").write_text("# Changelog\n\n- Previous → old rule.", "utf-8")

    assert {c.title for c in chunk_corpus(tmp_path)} == {"Imp"}
