"""Retrieval guided by the roles and glossary terms named in the question.

For a question: detect roles and glossary terms (entities.EntityMatcher), put the definitions of the detected terms
in the context, score every chunk with the dense embeddings, give a bonus to the chunks of the pages of the detected
roles (fusion), then fill the rest of the token budget with the top of the fused ranking (budget.fill_context).

    python -m botc.retrieval.guided "Can the Poisoner poison the Drunk?"
"""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from botc.retrieval.budget import fill_context, hf_token_counter
from botc.retrieval.contracts import (
    Boost,
    ChunkId,
    ChunkText,
    DenseScore,
    GlossaryEntry,
    GuidedConfig,
    Mass,
    MassFloor,
    Question,
    Retrieval,
    Temperature,
    TokenCount,
    TokenCounter,
)
from botc.retrieval.embedder import Embedder, SentenceTransformerEmbedder
from botc.retrieval.entities import EntityMatcher, parse_glossary, role_chunks
from botc.retrieval.fusion import bonus, fuse, rank
from botc.retrieval.index import build_store, load_index
from botc.retrieval.store import VectorStore

GLOSSARY_TITLE = "Glossary"  # its chunks are replaced by the definitions of the detected terms
DEFAULT_INDEX = Path("data/index/bge-small-en-v1.5_200w")
DEFAULT_GLOSSARY = Path("data/processed/wiki/Glossary.md")


class GuidedRetriever:
    """Retriever combining entity detection, dense scores, a bonus for the pages of the detected roles,
    and a token budget.

    Everything that does not depend on the question is prepared once, at construction: the pages of the roles,
    the two entity matchers, the text and the token cost of every chunk and of every definition."""

    def __init__(self, chunks: Sequence[Mapping[str, Any]],
                 store: VectorStore,
                 embedder: Embedder,
                 glossary: Sequence[GlossaryEntry],
                 count: TokenCounter,
                 config: GuidedConfig):
        """`chunks` are the lines of chunks.jsonl (id, title, character_type, header, text), the same chunks as in
        `store`; `embedder` must be the model that embedded them. `glossary` comes from entities.parse_glossary;
        `count` counts tokens with the tokenizer of the generator (budget.hf_token_counter).

        Raises ValueError if `chunks` is empty or if `store` does not hold exactly the ids of `chunks`.
        Hypothèse : roles are the pages whose character_type is not None (as in entities.role_chunks)."""
        if not chunks:
            raise ValueError("A guided retriever needs at least one chunk")
        ids = [ChunkId(chunk["id"]) for chunk in chunks]
        missing = [id for id in ids if id not in store]
        if missing or len(store) != len(ids):
            raise ValueError(f"The store holds {len(store)} chunks, not the {len(ids)} given "
                             f"(missing from the store: {missing[:5]})")
        self._store = store
        self._embedder = embedder
        self._count = count
        self._config = config
        self._pages = role_chunks([dict(chunk) for chunk in chunks])
        self._roles = EntityMatcher(list(self._pages)) if self._pages else None
        self._terms = EntityMatcher([entry.term for entry in glossary]) if glossary else None
        self._entries = {entry.term: entry for entry in glossary}
        self._texts = {ChunkId(chunk["id"]): ChunkText(f"{chunk['header']}\n\n{chunk['text']}") for chunk in chunks}
        self._costs = {id: count(text) for id, text in self._texts.items()}
        self._glossary_ids = {ChunkId(chunk["id"]) for chunk in chunks if chunk["title"] == GLOSSARY_TITLE}

    def retrieve(self, question: Question) -> Retrieval:
        """Returns the entities detected in the question and the context to give to the generator.

        Steps: roles and terms matched at the thresholds of the config; definitions of the detected terms (one per
        paragraph: synonyms share it), always included, their cost taken from the budget; dense scores of every
        chunk, chunks of the Glossary page removed; bonus for the chunks of the pages of the detected roles
        (fusion.bonus, fusion.fuse, fusion.rank); chunks = longest prefix of the ranking within the rest of the budget.
        If the definitions alone exceed the budget, no chunk is added and `tokens` is above the budget.
        Raises ValueError if the store returns fewer scores than it holds chunks (the bonus needs them all)."""
        roles = tuple(self._roles.match(question, self._config.threshold_roles)) if self._roles else ()
        terms = tuple(self._terms.match(question, self._config.threshold_terms)) if self._terms else ()
        definitions: list[GlossaryEntry] = []
        for match in terms:
            entry = self._entries[match.name]
            if entry.text not in {kept.text for kept in definitions}:
                definitions.append(entry)
        definition_tokens = sum(self._count(entry.text) for entry in definitions)

        query = self._embedder.embed([question])[0]
        hits = self._store.search(query, len(self._store))
        if len(hits) != len(self._store):
            raise ValueError(f"The store returned {len(hits)} scores for {len(self._store)} chunks")
        scores = {ChunkId(hit.id): DenseScore(hit.score) for hit in hits if hit.id not in self._glossary_ids}
        boosted = {id for match in roles for id in self._pages[match.name]}
        role_bonus = bonus(scores, boosted, self._config.rule)
        ranked = rank(fuse(scores, boosted, role_bonus))
        left = TokenCount(max(0, self._config.budget - definition_tokens))
        chunks = fill_context(ranked, self._costs, left)
        return Retrieval(question=question, roles=roles, terms=terms, definitions=tuple(definitions),
                         bonus=role_bonus, chunks=tuple(chunks),
                         tokens=TokenCount(definition_tokens + sum(self._costs[id] for id in chunks)))

    def text(self, id: ChunkId) -> ChunkText:
        """Returns the text of a chunk as put in the context: its header, a blank line, then its text.

        Raises KeyError for an unknown id."""
        return self._texts[id]


def main() -> None:
    """Prints, for a question, the detected entities, the definitions, the bonus and the chunks of the context.

    Loads the index (index.load_index, index.build_store), the embedder named in its manifest, the glossary and
    the Qwen3 tokenizer. Fusion rule from the command line, MassFloor(alpha=0.5, tau=0.05) by default."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY)
    parser.add_argument("--budget", type=int, default=2000, help="tokens for the definitions and the chunks")
    parser.add_argument("--alpha", type=float, default=0.5, help="minimal probability of the boosted chunks")
    parser.add_argument("--tau", type=float, default=0.05, help="temperature of the softmax")
    parser.add_argument("--cap", type=float, default=None, help="upper bound of the bonus")
    args = parser.parse_args()

    index = load_index(args.index)
    raw = cast(Any, index)  # Index types its chunks and manifest as bare dicts
    chunks = cast(list[dict[str, Any]], raw.chunks)
    model = cast(str, raw.manifest["model"])  # the question must be embedded by the model of the index
    rule = MassFloor(alpha=Mass(args.alpha), tau=Temperature(args.tau),
                     cap=None if args.cap is None else Boost(args.cap))
    retriever = GuidedRetriever(chunks, build_store(index), SentenceTransformerEmbedder(model),
                                parse_glossary(args.glossary), hf_token_counter(),
                                GuidedConfig(rule=rule, budget=TokenCount(args.budget)))
    found = retriever.retrieve(Question(args.question))

    print("roles:", ", ".join(f"{m.name} ({m.span!r}, {m.score:.2f})" for m in found.roles) or "-")
    print("terms:", ", ".join(f"{m.name} ({m.span!r}, {m.score:.2f})" for m in found.terms) or "-")
    print(f"bonus: {found.bonus:.4f}   tokens: {found.tokens} / {args.budget}\n")
    for entry in found.definitions:
        print(f"[definition] {entry.text}\n")
    for id in found.chunks:
        print(f"--- {id}")
        print(retriever.text(id)[:300], end="\n\n")


if __name__ == "__main__":
    main()
