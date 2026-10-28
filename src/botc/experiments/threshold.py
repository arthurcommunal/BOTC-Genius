"""Experiment to determine an efficient threshold for the entity detection.

Sentences of the wiki are matched at a low threshold t_min, every match is judged once, outside of this module
(by an LLM following docs/prompts/judge_entity_match.md), and precision, relative recall and F1 are then computed
for every threshold >= t_min without new judgments.

    python -m botc.experiments.threshold sample   # writes data/experiments/threshold/{roles,terms}/candidates.jsonl
    python -m botc.experiments.threshold curves   # reads the judgments.jsonl next to them, plots and prints the choice
"""

import argparse
import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from matplotlib.figure import Figure

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
    WordCount,
)
from botc.retrieval.contracts import (
    ChunkId,
    GlossaryTerm,
    Match,
    NormalizedText,
    Question,
    RoleName,
    Similarity,
)
from botc.retrieval.entities import EntityMatcher, parse_glossary, role_chunks
from botc.retrieval.index import CHUNKS_FILE

MIN_WORDS = WordCount(8)
MAX_WORDS = WordCount(40)
GLOSSARY_TITLE = "Glossary"                   # its sentences contain the term they define
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")  # same cut as corpus/chunk.py

DEFAULT_INDEX = Path("data/index/bge-small-en-v1.5_200w")
DEFAULT_GLOSSARY = Path("data/processed/wiki/Glossary.md")
DEFAULT_EXCHANGE = Path("data/experiments/threshold")         # candidates.jsonl and judgments.jsonl, per kind
DEFAULT_RESULTS = Path("src/botc/experiments/results/threshold")
CANDIDATES_FILE = "candidates.jsonl"
JUDGMENTS_FILE = "judgments.jsonl"
CURVE_FILE = "curve.jsonl"
THRESHOLD_STEP = 0.01
ABILITY_LINE = "\nAbility: "  # written in chunk headers by corpus/chunk.py

# Colors of the light chart surface (single series: no legend, the title names the measure).
SERIES_COLOR = "#2a78d6"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
SURFACE_COLOR = "#fcfcfb"


def sample_sentences(chunks: Sequence[Mapping[str, Any]],
                     n: int,
                     seed: Seed,
                     min_words: WordCount = MIN_WORDS,
                     max_words: WordCount = MAX_WORDS) -> list[SampledSentence]:
    """Draws n sentences from the field `text` of the chunks, reproducibly for a given seed.

    `chunks` are lines of chunks.jsonl; the header and the Glossary page are never read. A sentence is eligible
    if it has min_words to max_words words, bounds included. Ids go from 0 to n-1 in the order of the draw.
    Raises ValueError if there are fewer than n eligible sentences.
    Hypothèse : sentences are cut where ".", "!" or "?" is followed by whitespace (as in corpus/chunk.py);
    the draw is without replacement over occurrences, so a sentence repeated in two chunks can be drawn twice."""
    eligible = [(ChunkId(chunk["id"]), Sentence(text))
                for chunk in chunks if chunk["title"] != GLOSSARY_TITLE
                for text in SENTENCE_END.split(chunk["text"].strip())
                if min_words <= len(text.split()) <= max_words]
    if len(eligible) < n:
        raise ValueError(f"Only {len(eligible)} eligible sentences, {n} requested")
    drawn = random.Random(seed).sample(eligible, n)
    return [SampledSentence(id=SentenceId(i), chunk_id=chunk_id, text=text)
            for i, (chunk_id, text) in enumerate(drawn)]


def collect_candidates[N: (RoleName, GlossaryTerm)](sentences: Sequence[SampledSentence],
                                                    matcher: EntityMatcher[N],
                                                    t_min: Similarity) -> list[Candidate[N]]:
    """Applies the matcher to every sentence at the threshold t_min: one candidate per match.

    Ids go from 0 to m-1, sentence after sentence, then in the order of EntityMatcher.match (score, then name)."""
    candidates: list[Candidate[N]] = []
    for sentence in sentences:
        for match in matcher.match(Question(sentence.text), t_min):
            candidates.append(Candidate(id=CandidateId(len(candidates)), sentence=sentence, match=match))
    return candidates


def write_candidates[N: (RoleName, GlossaryTerm)](candidates: Sequence[Candidate[N]],
                                                  descriptions: Mapping[N, Description],
                                                  path: Path) -> None:
    """Writes the candidates to judge, one per line (one per match), in the format of experiments/contracts.py.

    Missing parent directories are created. Raises ValueError, before writing anything, if an entity
    has no description."""
    missing = sorted({c.match.name for c in candidates} - descriptions.keys())
    if missing:
        raise ValueError(f"No description for {missing}")
    lines = [json.dumps({"candidate_id": c.id, "sentence_id": c.sentence.id, "chunk_id": c.sentence.chunk_id,
                         "sentence": c.sentence.text, "span": c.match.span, "entity": c.match.name,
                         "score": c.match.score, "description": descriptions[c.match.name]}, ensure_ascii=False)
             for c in candidates]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), "utf-8")


def read_judgments[N: (RoleName, GlossaryTerm)](path: Path, candidates: Sequence[Candidate[N]]) -> list[bool]:
    """Returns the judgments of judgments.jsonl, aligned on the candidates (by candidate_id, whatever the file order).

    Raises ValueError if a candidate is not judged, is judged twice, if an id is unknown, or if a line
    has no int `candidate_id` or no bool `correct`. Blank lines are skipped."""
    known = {c.id for c in candidates}
    labels: dict[CandidateId, bool] = {}
    for number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        judgment = json.loads(line)
        candidate_id, correct = judgment.get("candidate_id"), judgment.get("correct")
        if type(candidate_id) is not int or type(correct) is not bool:
            raise ValueError(f"{path}:{number}: need an int candidate_id and a bool correct, got {line!r}")
        if candidate_id not in known:
            raise ValueError(f"{path}:{number}: unknown candidate {candidate_id}")
        if candidate_id in labels:
            raise ValueError(f"{path}:{number}: candidate {candidate_id} judged twice")
        labels[CandidateId(candidate_id)] = correct
    missing = sorted(known - labels.keys())
    if missing:
        raise ValueError(f"{len(missing)} candidates not judged in {path}, first ones: {missing[:10]}")
    return [labels[c.id] for c in candidates]


def pr_curve[N: (RoleName, GlossaryTerm)](candidates: Sequence[Candidate[N]],
                                          labels: Sequence[bool],
                                          thresholds: Sequence[Similarity]) -> list[CurvePoint]:
    """Precision, relative recall and F1 at every threshold, one point per threshold in the given order.

    Thresholds are inclusive. Relative recall divides by the number of correct candidates, i.e. the correct
    matches at t_min. Without any match at a threshold, precision and F1 are None.
    Raises ValueError if labels and candidates have different lengths or if no candidate is correct."""
    if len(labels) != len(candidates):
        raise ValueError(f"{len(labels)} labels for {len(candidates)} candidates")
    total_correct = sum(labels)
    if total_correct == 0:
        raise ValueError("No correct candidate: relative recall is undefined")
    curve: list[CurvePoint] = []
    for threshold in thresholds:
        kept = [label for candidate, label in zip(candidates, labels) if candidate.match.score >= threshold]
        matches, correct = len(kept), sum(kept)
        recall = Recall(correct / total_correct)
        precision = Precision(correct / matches) if matches else None
        f1 = None if precision is None else F1(_harmonic_mean(precision, recall))
        curve.append(CurvePoint(threshold=threshold, matches=matches, correct=correct,
                                precision=precision, recall=recall, f1=f1))
    return curve


def _harmonic_mean(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def best_threshold(curve: Sequence[CurvePoint]) -> CurvePoint:
    """Returns the point of maximal F1; at equal F1, the one of highest threshold (fewer false positives).

    Points without F1 (no match) are ignored. Raises ValueError if no point has an F1."""
    scored = [(point.f1, point.threshold, point) for point in curve if point.f1 is not None]
    if not scored:
        raise ValueError("No point of the curve has an F1")
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def plot_curves(curve: Sequence[CurvePoint], out_dir: Path) -> None:
    """Writes precision.png, recall.png and f1.png to out_dir (created if missing), each against the threshold.

    Axes span [0, 1]. A point without precision or F1 is a gap in the line, not a 0.
    The F1 figure marks the threshold chosen by best_threshold, when there is one."""
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [point.threshold for point in curve]
    measures: dict[str, tuple[str, list[float | None]]] = {
        "precision": ("Precision", [point.precision for point in curve]),
        "recall": ("Relative recall", [point.recall for point in curve]),
        "f1": ("F1", [point.f1 for point in curve]),
    }
    for file_name, (title, values) in measures.items():
        # Some members of the matplotlib stubs are partially unknown: figure and axes are used through Any.
        figure = cast(Any, Figure(figsize=(6, 4), facecolor=SURFACE_COLOR))
        axes = figure.add_subplot()
        axes.set_facecolor(SURFACE_COLOR)
        axes.plot(thresholds, [float("nan") if v is None else v for v in values],
                  color=SERIES_COLOR, linewidth=2)
        if file_name == "f1" and any(v is not None for v in values):
            best = best_threshold(curve)
            axes.axvline(best.threshold, color=AXIS_COLOR, linewidth=1, linestyle="--")
            axes.annotate(f"t = {best.threshold:.2f}, F1 = {best.f1:.2f}", (best.threshold, best.f1 or 0.0),
                          xytext=(6, 6), textcoords="offset points", color=TEXT_COLOR, fontsize=9)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1.02)
        axes.set_title(title, color=TEXT_COLOR, loc="left")
        axes.set_xlabel("Threshold", color=TEXT_COLOR)
        axes.grid(color=GRID_COLOR, linewidth=0.5)
        axes.tick_params(colors=TEXT_COLOR)
        for side in ["top", "right"]:
            axes.spines[side].set_visible(False)
        for side in ["bottom", "left"]:
            axes.spines[side].set_color(AXIS_COLOR)
        figure.savefig(out_dir / f"{file_name}.png", dpi=150, bbox_inches="tight")


def _read_chunks(index: Path) -> list[dict[str, Any]]:
    lines = (index / CHUNKS_FILE).read_text("utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _read_candidates[N: (RoleName, GlossaryTerm)](path: Path, name: Callable[[str], N]) -> list[Candidate[N]]:
    """Rebuilds the candidates written by write_candidates, so that `curves` depends only on the files."""
    candidates: list[Candidate[N]] = []
    for line in path.read_text("utf-8").splitlines():
        row = json.loads(line)
        sentence = SampledSentence(id=SentenceId(row["sentence_id"]), chunk_id=ChunkId(row["chunk_id"]),
                                   text=Sentence(row["sentence"]))
        match = Match(name=name(row["entity"]), span=NormalizedText(row["span"]), score=Similarity(row["score"]))
        candidates.append(Candidate(id=CandidateId(row["candidate_id"]), sentence=sentence, match=match))
    return candidates


def _role_description(role: RoleName, page: Sequence[Mapping[str, Any]]) -> Description:
    """Name, character type and ability: 'Librarian (Townsfolk): "You start knowing..."'.

    The ability is the "Ability: " line of a header (corpus/chunk.py adds it to every chunk but the Summary),
    else the first paragraph of the first chunk if it is quoted (one-chunk pages), else it is left out."""
    abilities = [chunk["header"].split(ABILITY_LINE, 1)[1] for chunk in page if ABILITY_LINE in chunk["header"]]
    first_paragraph = page[0]["text"].split("\n\n", 1)[0]
    ability = abilities[0] if abilities else first_paragraph if first_paragraph.startswith('"') else None
    name = f"{role} ({page[0]['character_type']})"
    return Description(f"{name}: {ability}" if ability else name)


def _sample(args: argparse.Namespace) -> None:
    chunks = _read_chunks(args.index)
    glossary = parse_glossary(args.glossary)
    pages = role_chunks(chunks)
    by_id = {chunk["id"]: chunk for chunk in chunks}
    role_descriptions = {role: _role_description(role, [by_id[id] for id in ids]) for role, ids in pages.items()}
    term_descriptions = {entry.term: Description(entry.text) for entry in glossary}
    sentences = sample_sentences(chunks, args.n, Seed(args.seed))
    t_min = Similarity(args.t_min)
    roles = collect_candidates(sentences, EntityMatcher(list(pages)), t_min)
    terms = collect_candidates(sentences, EntityMatcher(list(term_descriptions)), t_min)
    write_candidates(roles, role_descriptions, args.exchange / "roles" / CANDIDATES_FILE)
    write_candidates(terms, term_descriptions, args.exchange / "terms" / CANDIDATES_FILE)
    print(f"{len(sentences)} sentences: {len(roles)} role candidates, {len(terms)} term candidates "
          f"written to {args.exchange}")


def _curves(args: argparse.Namespace) -> None:
    steps = round((1 - args.t_min) / THRESHOLD_STEP)
    thresholds = [Similarity(round(args.t_min + i * THRESHOLD_STEP, 2)) for i in range(steps + 1)]
    _curves_of(args, "roles", RoleName, thresholds)
    _curves_of(args, "terms", GlossaryTerm, thresholds)


def _curves_of[N: (RoleName, GlossaryTerm)](args: argparse.Namespace, kind: str, name: Callable[[str], N],
                                            thresholds: list[Similarity]) -> None:
    candidates = _read_candidates(args.exchange / kind / CANDIDATES_FILE, name)
    labels = read_judgments(args.exchange / kind / JUDGMENTS_FILE, candidates)
    curve = pr_curve(candidates, labels, thresholds)
    out_dir: Path = args.results / kind
    plot_curves(curve, out_dir)
    (out_dir / CURVE_FILE).write_text("".join(json.dumps(vars(point)) + "\n" for point in curve), "utf-8")
    best = best_threshold(curve)
    print(f"{kind}: threshold {best.threshold:.2f}, precision {best.precision:.3f}, "
          f"relative recall {best.recall:.3f}, F1 {best.f1:.3f} ({best.correct}/{best.matches} correct)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exchange", type=Path, default=DEFAULT_EXCHANGE, help="candidates and judgments files")
    parser.add_argument("--t-min", type=float, default=0.6, help="lowest threshold, the one of the judged matches")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="draw sentences and write the candidates to judge")
    sample.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    sample.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY)
    sample.add_argument("--n", type=int, default=200)
    sample.add_argument("--seed", type=int, default=0)
    sample.set_defaults(run=_sample)
    curves = commands.add_parser("curves", help="read the judgments, plot the curves and print the best threshold")
    curves.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    curves.set_defaults(run=_curves)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
