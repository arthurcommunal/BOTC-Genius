import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from botc.experiments.contracts import (
    F1,
    Candidate,
    CandidateId,
    CurvePoint,
    Description,
    Precision,
    Recall,
    SampledSentence,
    Seed,
    Sentence,
    SentenceId,
)
from botc.experiments.threshold import (
    best_threshold,
    collect_candidates,
    plot_curves,
    pr_curve,
    read_judgments,
    sample_sentences,
    write_candidates,
)
from botc.retrieval.contracts import (
    ChunkId,
    Match,
    NormalizedText,
    RoleName,
    Similarity,
)
from botc.retrieval.entities import EntityMatcher

IMP, MONK, POISONER = RoleName("Imp"), RoleName("Monk"), RoleName("Poisoner")
DESCRIPTIONS = {IMP: Description("Imp (Demon) Ability: Each night*, choose a player: they die."),
                MONK: Description("Monk (Townsfolk) Ability: Each night*, choose a player (not yourself): "
                                  "they are safe from the Demon tonight.")}

# Five candidates for the curves: scores 0.4 to 0.8, judged [no, yes, no, yes, yes] (3 correct in all).
SCORES = [0.4, 0.5, 0.6, 0.7, 0.8]
LABELS = [False, True, False, True, True]


def chunk(id: str, text: str, title: str = "Imp", header: str = "") -> dict[str, Any]:
    """A line of chunks.jsonl, reduced to the fields read by sample_sentences."""
    return {"id": id, "title": title, "character_type": "Demon", "header": header, "text": text}


def sentence(n_words: int, word: str) -> str:
    """A sentence of exactly n_words words, recognizable by its repeated word."""
    return " ".join([word] * (n_words - 1) + ["end."])


def sampled(id: int, text: str, chunk_id: str = "Imp#0") -> SampledSentence:
    return SampledSentence(id=SentenceId(id), chunk_id=ChunkId(chunk_id), text=Sentence(text))


def candidate(id: int, score: float, name: RoleName = IMP, sentence_id: int = 0) -> Candidate[RoleName]:
    match = Match(name=name, span=NormalizedText(name.lower()), score=Similarity(score))
    return Candidate(id=CandidateId(id), sentence=sampled(sentence_id, f"The {name} acts."), match=match)


def scored() -> list[Candidate[RoleName]]:
    return [candidate(i, score) for i, score in enumerate(SCORES)]


def point(threshold: float, f1: float | None) -> CurvePoint:
    """A valid curve point with the given F1; None means no match at this threshold."""
    if f1 is None:
        return CurvePoint(threshold=Similarity(threshold), matches=0, correct=0, precision=None,
                          recall=Recall(0.0), f1=None)
    return CurvePoint(threshold=Similarity(threshold), matches=1, correct=1, precision=Precision(1.0),
                      recall=Recall(f1), f1=F1(f1))


def write_lines(path: Path, lines: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), "utf-8")


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


# --- sample_sentences ---

def test_sampling_is_reproducible_with_the_same_seed():
    """Calibration must be rerun on the same sentences, so the draw depends on the seed only.

    Two calls with the same chunks, n and seed return the same sentences, with the same ids, in the same order."""
    chunks = [chunk(f"Imp#{i}", sentence(10, f"w{i}")) for i in range(20)]
    first = sample_sentences(chunks, n=5, seed=Seed(7))
    assert first == sample_sentences(chunks, n=5, seed=Seed(7))


def test_sentences_are_cut_after_a_full_stop_an_exclamation_or_a_question_mark():
    """A chunk text holds several sentences: each one is a candidate on its own.

    The text is cut where ".", "!" or "?" is followed by whitespace; the punctuation stays with its sentence."""
    first, second, third = sentence(8, "a")[:-1] + "!", sentence(8, "b")[:-1] + "?", sentence(8, "c")
    chunks = [chunk("Imp#0", f"{first} {second}\n{third}")]
    assert {s.text for s in sample_sentences(chunks, n=3, seed=Seed(0))} == {first, second, third}


def test_word_bounds_are_inclusive():
    """By default a sentence is kept if it has 8 to 40 words, bounds included.

    Out of sentences of 7, 8, 40 and 41 words, only the ones of 8 and 40 words can be drawn."""
    text = " ".join([sentence(7, "a"), sentence(8, "b"), sentence(40, "c"), sentence(41, "d")])
    drawn = sample_sentences([chunk("Imp#0", text)], n=2, seed=Seed(0))
    assert {s.text for s in drawn} == {sentence(8, "b"), sentence(40, "c")}


def test_the_header_is_never_sampled():
    """The header contains the page title, which would make every role page match its own role.

    Only the `text` field is read: a valid sentence of the header is never drawn."""
    chunks = [chunk("Imp#0", sentence(10, "text"), header=sentence(10, "header"))]
    assert [s.text for s in sample_sentences(chunks, n=1, seed=Seed(0))] == [sentence(10, "text")]


def test_the_glossary_page_is_never_sampled():
    """Every sentence of the glossary contains the term it defines, which would inflate precision.

    Chunks whose title is "Glossary" are skipped."""
    chunks = [chunk("Glossary#0", sentence(10, "glossary"), title="Glossary"), chunk("Imp#0", sentence(10, "imp"))]
    assert [s.text for s in sample_sentences(chunks, n=1, seed=Seed(0))] == [sentence(10, "imp")]


def test_sampled_sentences_are_numbered_and_keep_their_chunk():
    """Each sentence gets an id from 0 to n-1 in the order of the draw, and the id of its chunk.

    The chunk id lets the judge and Arthur go back to the source."""
    chunks = [chunk("Imp#0", sentence(10, "imp")), chunk("Monk#2", sentence(10, "monk"), title="Monk")]
    drawn = sample_sentences(chunks, n=2, seed=Seed(0))
    assert [s.id for s in drawn] == [0, 1]
    assert {s.text: s.chunk_id for s in drawn} == {sentence(10, "imp"): "Imp#0", sentence(10, "monk"): "Monk#2"}


def test_sampling_more_sentences_than_eligible_raises():
    """Fewer eligible sentences than n means the sample cannot have the requested size.

    ValueError rather than a silently smaller sample."""
    with pytest.raises(ValueError):
        sample_sentences([chunk("Imp#0", sentence(10, "imp"))], n=2, seed=Seed(0))


# --- collect_candidates ---

def test_candidates_are_numbered_sentence_by_sentence_then_in_match_order():
    """One candidate per match: a sentence with several matches gives several candidates, one without gives none.

    Ids go from 0 to m-1, sentence after sentence, then in the order returned by the matcher (score, then name).
    Each candidate carries its sampled sentence and the match itself."""
    matcher = EntityMatcher([IMP, MONK, POISONER])
    sentences = [sampled(0, "The Imp kills at night."), sampled(1, "Nothing to see here."),
                 sampled(2, "The Monk protects the Imp.")]
    candidates = collect_candidates(sentences, matcher, t_min=Similarity(0.99))
    assert [(c.id, c.sentence, c.match.name) for c in candidates] == [
        (0, sentences[0], IMP), (1, sentences[2], IMP), (2, sentences[2], MONK)]
    assert candidates[2].match == Match(name=MONK, span=NormalizedText("monk"), score=Similarity(1.0))


def test_t_min_is_the_matching_threshold():
    """t_min is the threshold given to the matcher: matches below it are not collected.

    "poisoned" matches Poisoner below 1: collected at t_min = 0.5, not at t_min = 0.99."""
    matcher = EntityMatcher([IMP, MONK, POISONER])
    sentences = [sampled(0, "The Imp was poisoned.")]
    assert [c.match.name for c in collect_candidates(sentences, matcher, t_min=Similarity(0.99))] == [IMP]
    assert [c.match.name for c in collect_candidates(sentences, matcher, t_min=Similarity(0.5))] == [IMP, POISONER]


# --- write_candidates ---

def test_a_candidate_line_holds_everything_the_judge_needs(tmp_path: Path):
    """The judging session reads only candidates.jsonl, so each line is self-contained.

    It holds the ids, the source chunk, the sentence as written, the normalized span, the entity, the score
    and the description of the entity."""
    path = tmp_path / "candidates.jsonl"
    write_candidates([candidate(0, 1.0)], DESCRIPTIONS, path)
    assert read_lines(path) == [{"candidate_id": 0, "sentence_id": 0, "chunk_id": "Imp#0", "sentence": "The Imp acts.",
                                 "span": "imp", "entity": "Imp", "score": 1.0, "description": DESCRIPTIONS[IMP]}]


def test_one_line_per_candidate_in_id_order(tmp_path: Path):
    """A sentence with several matches is written once per match.

    Lines follow the order of the candidates: candidate_id 0, 1, 2; the sentence id repeats."""
    path = tmp_path / "candidates.jsonl"
    write_candidates([candidate(0, 1.0, IMP, 0), candidate(1, 1.0, MONK, 0), candidate(2, 0.9, IMP, 1)],
                     DESCRIPTIONS, path)
    assert [(line["candidate_id"], line["sentence_id"]) for line in read_lines(path)] == [(0, 0), (1, 0), (2, 1)]


def test_write_candidates_creates_missing_directories(tmp_path: Path):
    """The output goes to data/experiments/threshold/{roles,terms}/, which may not exist yet.

    Missing parent directories are created."""
    path = tmp_path / "threshold" / "roles" / "candidates.jsonl"
    write_candidates([candidate(0, 1.0)], DESCRIPTIONS, path)
    assert path.exists()


def test_an_entity_without_description_raises(tmp_path: Path):
    """The judge cannot decide without the ability or the definition of the entity.

    ValueError if an entity of the candidates is missing from the descriptions."""
    with pytest.raises(ValueError):
        write_candidates([candidate(0, 0.9, POISONER)], DESCRIPTIONS, tmp_path / "candidates.jsonl")


# --- read_judgments ---

def test_judgments_are_aligned_on_the_candidates(tmp_path: Path):
    """The judging session may write its judgments in any order.

    The labels come back in the order of the candidates, matched by candidate_id."""
    path = tmp_path / "judgments.jsonl"
    write_lines(path, [{"candidate_id": 2, "correct": True}, {"candidate_id": 0, "correct": False},
                       {"candidate_id": 1, "correct": True}])
    assert read_judgments(path, [candidate(0, 1.0), candidate(1, 0.9), candidate(2, 0.8)]) == [False, True, True]


@pytest.mark.parametrize("judgments", [
    [{"candidate_id": 0, "correct": True}],
    [{"candidate_id": 0, "correct": True}, {"candidate_id": 1, "correct": True}, {"candidate_id": 1, "correct": False}],
    [{"candidate_id": 0, "correct": True}, {"candidate_id": 1, "correct": True}, {"candidate_id": 5, "correct": True}],
    [{"candidate_id": 0, "correct": True}, {"candidate_id": 1, "correct": "yes"}],
], ids=["missing", "duplicate", "unknown id", "not a bool"])
def test_an_incomplete_or_malformed_judgments_file_raises(tmp_path: Path, judgments: list[dict[str, Any]]):
    """A label that is missing, given twice, about an unknown candidate or not a bool would bias the curves silently.

    ValueError in each case, with two candidates (ids 0 and 1)."""
    path = tmp_path / "judgments.jsonl"
    write_lines(path, judgments)
    with pytest.raises(ValueError):
        read_judgments(path, [candidate(0, 1.0), candidate(1, 0.9)])


# --- pr_curve ---

def test_curve_points_count_matches_and_correct_ones_above_each_threshold():
    """At threshold t: precision = correct >= t / matches >= t, relative recall = correct >= t / all correct,
    F1 = their harmonic mean. Thresholds are inclusive (0.6 keeps the score 0.6).

    Scores 0.4..0.8 judged [no, yes, no, yes, yes]: t = 0.4 → 5 matches, 3 correct; t = 0.6 → 3, 2; t = 0.75 → 1, 1."""
    curve = pr_curve(scored(), LABELS, [Similarity(0.4), Similarity(0.6), Similarity(0.75)])
    assert [(p.threshold, p.matches, p.correct) for p in curve] == [(0.4, 5, 3), (0.6, 3, 2), (0.75, 1, 1)]
    assert [p.precision for p in curve] == pytest.approx([0.6, 2 / 3, 1.0])
    assert [p.recall for p in curve] == pytest.approx([1.0, 2 / 3, 1 / 3])
    assert [p.f1 for p in curve] == pytest.approx([0.75, 2 / 3, 0.5])


def test_curve_points_follow_the_order_of_the_thresholds():
    """One point per threshold, in the order given: thresholds are not sorted."""
    curve = pr_curve(scored(), LABELS, [Similarity(0.75), Similarity(0.4)])
    assert [p.threshold for p in curve] == [0.75, 0.4]


def test_a_threshold_without_match_has_no_precision_nor_f1():
    """Above the best score there is no match: precision is 0/0, undefined, so None, and so is F1.

    The relative recall is 0: no correct match is kept."""
    [above] = pr_curve(scored(), LABELS, [Similarity(0.9)])
    assert (above.matches, above.correct, above.precision, above.recall, above.f1) == (0, 0, None, 0.0, None)


def test_f1_is_zero_when_only_wrong_matches_are_kept():
    """With matches but no correct one, precision and recall are 0: F1 is 0, not a division by zero.

    Scores 0.5 (correct) and 0.9 (wrong), threshold 0.8."""
    [high] = pr_curve([candidate(0, 0.5), candidate(1, 0.9)], [True, False], [Similarity(0.8)])
    assert (high.precision, high.recall, high.f1) == (0.0, 0.0, 0.0)


def test_labels_and_candidates_of_different_lengths_raise():
    """Labels are aligned on candidates: a length mismatch means they do not come from the same run."""
    with pytest.raises(ValueError):
        pr_curve(scored(), LABELS[:-1], [Similarity(0.4)])


def test_a_curve_without_any_correct_candidate_raises():
    """Relative recall divides by the number of correct candidates: with none, it is undefined.

    ValueError rather than a curve of undefined recalls."""
    with pytest.raises(ValueError):
        pr_curve(scored(), [False] * len(SCORES), [Similarity(0.4)])


# --- best_threshold ---

def test_the_best_threshold_maximizes_f1():
    """The chosen threshold is the point of maximal F1, whatever its position in the curve."""
    assert best_threshold([point(0.4, 0.6), point(0.5, 0.8), point(0.6, 0.7)]).threshold == 0.5


def test_at_equal_f1_the_highest_threshold_wins():
    """For the same F1, a higher threshold keeps fewer false positives.

    Points given in increasing and in decreasing order of threshold."""
    assert best_threshold([point(0.4, 0.8), point(0.5, 0.8)]).threshold == 0.5
    assert best_threshold([point(0.5, 0.8), point(0.4, 0.8)]).threshold == 0.5


def test_points_without_f1_are_ignored():
    """A threshold without any match has no F1: it can never be chosen."""
    assert best_threshold([point(0.4, 0.5), point(0.9, None)]).threshold == 0.4


@pytest.mark.parametrize("curve", [[], [point(0.9, None), point(1.0, None)]], ids=["empty", "no match"])
def test_a_curve_without_any_f1_raises(curve: list[CurvePoint]):
    """Without any point having an F1 (empty curve, or no match at any threshold), there is nothing to choose.

    ValueError."""
    with pytest.raises(ValueError):
        best_threshold(curve)


# --- plot_curves ---

def test_plot_curves_writes_three_figures(tmp_path: Path):
    """Precision, relative recall and F1 are each plotted against the threshold, in their own PNG file.

    The output directory is created if missing; it then contains exactly precision.png, recall.png and f1.png.
    A point without precision (no match) does not prevent plotting."""
    out_dir = tmp_path / "results" / "threshold" / "roles"
    plot_curves(pr_curve(scored(), LABELS, [Similarity(t) for t in [0.4, 0.6, 0.75, 0.9]]), out_dir)
    assert sorted(p.name for p in out_dir.iterdir()) == ["f1.png", "precision.png", "recall.png"]
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in out_dir.iterdir())


# --- contracts ---

@pytest.mark.parametrize("fields", [{"id": SentenceId(-1)}, {"text": Sentence("  ")}], ids=["negative id", "blank"])
def test_an_invalid_sampled_sentence_raises(fields: dict[str, Any]):
    """A sampled sentence has an id >= 0 and a non-blank text."""
    with pytest.raises(ValueError):
        dataclasses.replace(sampled(0, "The Imp acts."), **fields)


def test_a_candidate_id_is_not_negative():
    """Candidate ids are line numbers of candidates.jsonl: from 0."""
    with pytest.raises(ValueError):
        dataclasses.replace(candidate(0, 1.0), id=CandidateId(-1))


@pytest.mark.parametrize("fields", [
    {"correct": 3},
    {"correct": -1},
    {"precision": None, "f1": None},
    {"matches": 0, "correct": 0},
    {"f1": None},
    {"precision": Precision(1.5)},
    {"recall": Recall(-0.1)},
    {"f1": F1(1.2)},
], ids=["correct > matches", "negative correct", "no precision with matches", "precision without match",
        "f1 None alone", "precision > 1", "negative recall", "f1 > 1"])
def test_an_inconsistent_curve_point_raises(fields: dict[str, Any]):
    """A curve point has 0 <= correct <= matches, a precision exactly when matches > 0, an F1 exactly when it has
    a precision, and precision, recall and F1 in [0, 1].

    Each case changes one valid point (2 matches, 1 correct, precision 0.5, recall 1, F1 2/3)."""
    valid = CurvePoint(threshold=Similarity(0.5), matches=2, correct=1, precision=Precision(0.5),
                       recall=Recall(1.0), f1=F1(2 / 3))
    with pytest.raises(ValueError):
        dataclasses.replace(valid, **fields)


def test_a_point_without_match_is_valid():
    """No match at a threshold is a normal case, not an error: precision and F1 are None, recall is 0."""
    assert point(0.9, None).precision is None


def test_a_candidate_is_immutable():
    """Candidates are shared between the written file and the curves: they cannot change in between."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate(0, 1.0).id = CandidateId(1)  # type: ignore[misc]
