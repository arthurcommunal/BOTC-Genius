import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from botc.retrieval.contracts import (
    GlossaryTerm,
    NormalizedText,
    Question,
    RoleName,
    Similarity,
)
from botc.retrieval.entities import (
    EntityMatcher,
    GlossaryEntry,
    Match,
    normalize,
    parse_glossary,
    role_chunks,
)

ROLES = [RoleName(name) for name in ["Imp", "Poisoner", "Drunk", "Monk", "Mayor", "Spy", "Fortune Teller", "Pit-Hag",
                                     "Scarlet Woman", "Summoner", "Po"]]
TERMS = [GlossaryTerm(term) for term in ["Each night", "Each night*", "In play", "Not in play", "Demon", "Demon, The",
                                         "Poisoned", "Drunk"]]

GLOSSARY_MD = """# Glossary

Ability: The special power of a character.

Choose: The player decides: this or that.

Demon, The: The player that has the Demon character.

Each night*: Every night except the first.

Resurrected / Reborn: A dead player becoming alive again.
"""


# --- normalize ---

def test_normalize_lowercases_and_collapses_whitespace():
    """Case and spacing carry no meaning for matching.

    Upper case becomes lower case; tabs, newlines and repeated spaces become a single space;
    leading and trailing spaces are removed."""
    assert normalize("  Fortune \t Teller\n") == "fortune teller"


def test_normalize_drops_possessives():
    """A possessive points to the same entity as the bare name.

    "'s" at the end of a word is removed, with a straight or a typographic apostrophe;
    the apostrophe of a plural possessive is removed as well."""
    assert normalize("Imp's") == "imp"
    assert normalize("Imp’s") == "imp"
    assert normalize("players'") == "players"


def test_normalize_turns_punctuation_into_spaces():
    """Hyphens, apostrophes, parentheses and question marks separate words.

    "Pit-Hag" and "pit hag" get the same form, and so do the wiki name and a typed question."""
    assert normalize("Pit-Hag") == "pit hag"
    assert normalize("Lil' Monsta") == "lil monsta"
    assert normalize("God of Ug (Ug Mode)") == "god of ug ug mode"
    assert normalize("Can the Spy register?") == "can the spy register"


def test_normalize_keeps_the_star():
    """The star is BOTC vocabulary: "Each night*" means every night except the first.

    It is kept, attached to its word, so that "Each night*" and "Each night" stay different."""
    assert normalize("Each night*") == "each night*"


def test_normalize_strips_accents():
    """Accented letters become their unaccented letter, so that a typed question without accents matches."""
    assert normalize("Déjà vu") == "deja vu"


def test_normalize_of_nothing_meaningful_is_empty():
    """An empty text, or a text made only of punctuation, normalizes to the empty string."""
    assert normalize("") == ""
    assert normalize("?!") == ""


@given(st.text())
def test_normalize_is_idempotent(text: str):
    """Normalizing an already normalized text changes nothing (for any text, hypothesis)."""
    once = normalize(text)
    assert normalize(once) == once


@given(st.text())
def test_normalize_output_is_lowercase_words_separated_by_single_spaces(text: str):
    """For any text, the output only contains a-z, 0-9 and "*", in words separated by single spaces.

    No leading or trailing space; the empty string is allowed."""
    assert re.fullmatch(r"([a-z0-9*]+( [a-z0-9*]+)*)?", normalize(text))


# --- parse_glossary ---

@pytest.fixture
def glossary_path(tmp_path: Path) -> Path:
    path = tmp_path / "Glossary.md"
    path.write_text(GLOSSARY_MD, "utf-8")
    return path


def test_glossary_entries_follow_the_file_order_and_skip_the_title(glossary_path: Path):
    """One entry per "Term: definition" paragraph, in the order of the file.

    The "# Glossary" title and the blank lines are not entries."""
    terms = [entry.term for entry in parse_glossary(glossary_path)]
    assert terms == ["Ability", "Choose", "Demon, The", "Each night*", "Resurrected", "Reborn"]


def test_glossary_text_is_the_whole_paragraph(glossary_path: Path):
    """The text of an entry is the full paragraph, term included, as written in the Markdown.

    It is what will be put in the context of the LLM."""
    assert parse_glossary(glossary_path)[0] == GlossaryEntry(
        term=GlossaryTerm("Ability"), text="Ability: The special power of a character.")


def test_the_term_stops_at_the_first_colon(glossary_path: Path):
    """A colon inside the definition does not belong to the term.

    "Choose: The player decides: this or that." gives the term "Choose"."""
    entry = parse_glossary(glossary_path)[1]
    assert entry == GlossaryEntry(term=GlossaryTerm("Choose"), text="Choose: The player decides: this or that.")


def test_the_term_is_kept_raw(glossary_path: Path):
    """The term is not normalized: commas and stars stay ("Demon, The", "Each night*").

    Normalization is the job of the matcher; the term is the key used to find the entry back."""
    terms = [entry.term for entry in parse_glossary(glossary_path)]
    assert "Demon, The" in terms
    assert "Each night*" in terms


def test_a_paragraph_with_several_terms_gives_one_entry_per_term(glossary_path: Path):
    """"Resurrected / Reborn: ..." defines several synonyms at once.

    It gives one entry per term, split on " / ", all sharing the same full paragraph as text."""
    text = "Resurrected / Reborn: A dead player becoming alive again."
    assert parse_glossary(glossary_path)[-2:] == [GlossaryEntry(term=GlossaryTerm("Resurrected"), text=text),
                                                  GlossaryEntry(term=GlossaryTerm("Reborn"), text=text)]


def test_a_paragraph_without_colon_raises(tmp_path: Path):
    """A paragraph without "Term: " means the format of the glossary changed.

    Fail loudly with a ValueError rather than silently dropping or mislabelling a definition."""
    path = tmp_path / "Glossary.md"
    path.write_text("# Glossary\n\nAbility: A power.\n\nA paragraph without any term.\n", "utf-8")
    with pytest.raises(ValueError):
        parse_glossary(path)


@pytest.mark.parametrize("term", ["", "   "])
def test_a_glossary_entry_needs_a_term(term: str):
    """A glossary entry with an empty or blank term is invalid: ValueError at construction."""
    with pytest.raises(ValueError):
        GlossaryEntry(term=GlossaryTerm(term), text="Ability: A power.")


def test_a_glossary_entry_text_needs_the_term_separator():
    """The text of an entry is a "Term: definition" paragraph: without ": " it is invalid (ValueError)."""
    with pytest.raises(ValueError):
        GlossaryEntry(term=GlossaryTerm("Ability"), text="Ability, a power.")


# --- role_chunks ---

def chunk(id: str, title: str, character_type: str | None) -> dict[str, Any]:
    return {"id": id, "title": title, "character_type": character_type, "text": "..."}


def test_role_chunks_group_the_ids_by_role():
    """Each role (page with a character type) is mapped to the ids of the chunks of its page."""
    chunks = [chunk("Imp#0", "Imp", "Demon"), chunk("Drunk#0", "Drunk", "Outsider"),
              chunk("Imp#1", "Imp", "Demon"), chunk("Drunk#1", "Drunk", "Outsider")]
    assert role_chunks(chunks) == {"Imp": ["Imp#0", "Imp#1"], "Drunk": ["Drunk#0", "Drunk#1"]}


def test_role_chunks_keep_the_input_order():
    """The ids keep the order of the input (the page order), they are not sorted.

    A string sort would put "Imp#10" before "Imp#2"."""
    ids = [f"Imp#{i}" for i in range(12)]
    assert role_chunks([chunk(id, "Imp", "Demon") for id in ids]) == {"Imp": ids}


def test_rules_pages_are_not_roles():
    """Pages without a character type (Glossary, Rules Explanation...) are not in the result."""
    chunks = [chunk("Glossary#0", "Glossary", None), chunk("Imp#0", "Imp", "Demon")]
    assert role_chunks(chunks) == {"Imp": ["Imp#0"]}


def test_role_chunks_of_nothing_is_empty():
    assert role_chunks([]) == {}


# --- EntityMatcher ---

@pytest.fixture
def roles() -> EntityMatcher[RoleName]:
    return EntityMatcher(ROLES)


@pytest.fixture
def terms() -> EntityMatcher[GlossaryTerm]:
    return EntityMatcher(TERMS)


def test_an_exact_name_scores_one(roles: EntityMatcher[RoleName]):
    """A segment equal to a name (after normalization) has a cosine of 1.

    The match gives the name in its original form and the segment in its normalized form."""
    assert [(m.name, m.span, m.score) for m in roles.match(Question("Who is the Fortune Teller?"), threshold=Similarity(0.99))] == [
        ("Fortune Teller", "fortune teller", pytest.approx(1.0))]


def test_case_possessive_and_hyphen_variants_score_one(roles: EntityMatcher[RoleName]):
    """Variants removed by normalize are exact matches: "THE IMP'S", "pit hag" for "Pit-Hag"."""
    assert [(m.name, m.score) for m in roles.match(Question("THE IMP'S bluff"), threshold=Similarity(0.99))] == [
        ("Imp", pytest.approx(1.0))]
    assert [(m.name, m.score) for m in roles.match(Question("is the pit hag evil"), threshold=Similarity(0.99))] == [
        ("Pit-Hag", pytest.approx(1.0))]


def test_an_inflected_form_matches_its_entity_below_one(roles: EntityMatcher[RoleName]):
    """An inflected form ("poisoned") is close to its entity but not equal to it.

    Poisoner is the best match, with a score strictly between 0.5 and 1."""
    best = roles.match(Question("I was poisoned"), threshold=Similarity(0.0))[0]
    assert best.name == "Poisoner"
    assert 0.5 < best.score < 1


def test_an_entity_found_in_several_segments_is_returned_once(roles: EntityMatcher[RoleName]):
    """"fortune", "teller", "fortune teller", "the fortune teller" all point to Fortune Teller.

    It is returned once, with its best score."""
    matches = roles.match(Question("the fortune teller"), threshold=Similarity(0.5))
    assert [(m.name, m.score) for m in matches] == [("Fortune Teller", pytest.approx(1.0))]


def test_at_equal_score_the_shortest_segment_is_kept(roles: EntityMatcher[RoleName]):
    """A word with no n-gram in the vocabulary ("the") leaves the vector unchanged.

    "the fortune teller" and "fortune teller" then have the same score: the shortest segment is kept,
    it is the most precise one."""
    assert roles.match(Question("the fortune teller"), threshold=Similarity(0.5))[0].span == "fortune teller"


def test_scores_are_rounded_so_that_exact_matches_tie(roles: EntityMatcher[RoleName]):
    """Cosines are rounded to 6 decimals: two exact matches both score exactly 1.0.

    Without rounding, float errors (0.9999999999999997) would break the ties used for sorting."""
    assert [m.score for m in roles.match(Question("monk drunk"), threshold=Similarity(0.99))] == [1.0, 1.0]


def test_several_entities_in_one_question(roles: EntityMatcher[RoleName]):
    """A question about an interaction finds every role it names, and no other.

    "Can the Poisoner poison the Drunk?" gives Poisoner and Drunk."""
    matches = roles.match(Question("Can the Poisoner poison the Drunk?"), threshold=Similarity(0.7))
    assert {m.name for m in matches} == {"Poisoner", "Drunk"}


def test_segments_are_at_most_max_span_words_long():
    """With max_span_words=1, only single words are compared to the names.

    "fortune teller" is then never a segment: Fortune Teller is found through one word, below 1."""
    matches = EntityMatcher(ROLES, max_span_words=1).match(Question("the fortune teller"), threshold=Similarity(0.5))
    assert [m.name for m in matches] == ["Fortune Teller"]
    assert " " not in matches[0].span
    assert matches[0].score < 1


def test_the_star_distinguishes_each_night_from_each_night_star(terms: EntityMatcher[GlossaryTerm]):
    """"each night" and "each night*" are two different glossary terms, and each one matches exactly its own."""
    assert [m.name for m in terms.match(Question("each night"), threshold=Similarity(0.99))] == ["Each night"]
    assert [m.name for m in terms.match(Question("each night*"), threshold=Similarity(0.99))] == ["Each night*"]


def test_overlapping_segments_are_not_resolved(terms: EntityMatcher[GlossaryTerm]):
    """Overlaps are kept on purpose: definitions are cheap in tokens.

    "the demon" matches "Demon, The" (segment "the demon") and "Demon" (segment "demon")."""
    assert [m.name for m in terms.match(Question("the demon"), threshold=Similarity(0.99))] == ["Demon", "Demon, The"]


def test_a_question_without_known_ngrams_matches_nothing(roles: EntityMatcher[RoleName]):
    """A segment sharing no n-gram with any name has a score of 0, which never matches.

    It holds even with threshold=0: a match needs a strictly positive score."""
    assert roles.match(Question("xyz qqq"), threshold=Similarity(0.0)) == []


def test_an_empty_question_matches_nothing(roles: EntityMatcher[RoleName]):
    """An empty question, or one made only of punctuation, has no segment."""
    assert roles.match(Question(""), threshold=Similarity(0.0)) == []
    assert roles.match(Question("?!"), threshold=Similarity(0.0)) == []


def test_matches_are_sorted_by_decreasing_score(roles: EntityMatcher[RoleName]):
    """The best match comes first: "drunk" (exact, 1) before "poisoned" (Poisoner, below 1)."""
    assert [m.name for m in roles.match(Question("poisoned drunk"), threshold=Similarity(0.5))] == ["Drunk", "Poisoner"]


def test_equal_scores_are_sorted_by_name(roles: EntityMatcher[RoleName]):
    """At equal score, matches are sorted by name, whatever their order in the question."""
    assert [m.name for m in roles.match(Question("monk drunk"), threshold=Similarity(0.99))] == ["Drunk", "Monk"]


def test_the_threshold_is_inclusive(roles: EntityMatcher[RoleName]):
    """A match whose score equals the threshold is kept."""
    score = roles.match(Question("I was poisoned"), threshold=Similarity(0.0))[0].score
    assert "Poisoner" in [m.name for m in roles.match(Question("I was poisoned"), threshold=score)]


def test_a_higher_threshold_keeps_a_subset_of_the_matches(roles: EntityMatcher[RoleName]):
    """Raising the threshold can only remove matches, never add some."""
    question = Question("Can the Poisoner poison the Drunk or the fortune teller?")
    high = {m.name for m in roles.match(question, threshold=Similarity(0.9))}
    low = {m.name for m in roles.match(question, threshold=Similarity(0.5))}
    assert high <= low


def test_a_matcher_without_names_raises():
    """An EntityMatcher needs at least one name to build its vocabulary of n-grams."""
    with pytest.raises(ValueError):
        EntityMatcher[RoleName]([])


@pytest.mark.parametrize("settings", [{"ngram": 0}, {"max_span_words": 0}])
def test_a_matcher_needs_ngrams_and_segments_of_at_least_one(settings: dict[str, int]):
    """n-grams of 0 characters or segments of 0 words make no sense: ValueError at construction."""
    with pytest.raises(ValueError):
        EntityMatcher(ROLES, **settings)


# --- Match ---

@pytest.mark.parametrize("score", [0.0, -0.1, 1.000001])
def test_a_match_score_is_in_zero_one(score: float):
    """A match has a cosine in ]0, 1]: 0 means no shared n-gram, above 1 is not a cosine.

    Any other score is invalid: ValueError at construction."""
    with pytest.raises(ValueError):
        Match(name=RoleName("Imp"), span=NormalizedText("imp"), score=Similarity(score))


def test_a_score_of_one_is_a_valid_match():
    """The upper bound is included: an exact match scores 1."""
    assert Match(name=RoleName("Imp"), span=NormalizedText("imp"), score=Similarity(1.0)).score == 1.0


def test_a_match_needs_a_span():
    """A match comes from a segment of the question: an empty span is invalid (ValueError)."""
    with pytest.raises(ValueError):
        Match(name=RoleName("Imp"), span=NormalizedText(""), score=Similarity(1.0))


def test_a_match_is_immutable():
    """A match is a frozen dataclass: its fields cannot be reassigned after construction."""
    match = Match(name=RoleName("Imp"), span=NormalizedText("imp"), score=Similarity(1.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        match.score = 0.5  # type: ignore[misc]
