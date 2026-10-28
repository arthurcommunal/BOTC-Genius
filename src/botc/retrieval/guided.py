"""Retrieval guided by the roles and glossary terms named in the question.

For a question: detect roles and glossary terms (entities.EntityMatcher), put the definitions of the detected terms
in the context, score every chunk with the dense embeddings, give a bonus to the chunks of the pages of the detected
roles (fusion), then fill the rest of the token budget with the top of the fused ranking (budget.fill_context).

    python -m botc.retrieval.guided "Can the Poisoner poison the Drunk?"
"""

from collections.abc import Mapping, Sequence
from typing import Any

from botc.retrieval.contracts import (
    ChunkId,
    ChunkText,
    GlossaryEntry,
    GuidedConfig,
    Question,
    Retrieval,
    TokenCounter,
)
from botc.retrieval.embedder import Embedder
from botc.retrieval.store import VectorStore

GLOSSARY_TITLE = "Glossary"  # its chunks are replaced by the definitions of the detected terms


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
        raise NotImplementedError

    def retrieve(self, question: Question) -> Retrieval:
        """Returns the entities detected in the question and the context to give to the generator.

        Steps: roles and terms matched at the thresholds of the config; definitions of the detected terms (one per
        paragraph: synonyms share it), always included, their cost taken from the budget; dense scores of every
        chunk, chunks of the Glossary page removed; bonus for the chunks of the pages of the detected roles
        (fusion.bonus, fusion.fuse, fusion.rank); chunks = longest prefix of the ranking within the rest of the budget.
        If the definitions alone exceed the budget, no chunk is added and `tokens` is above the budget.
        Raises ValueError if the store returns fewer scores than it holds chunks (the bonus needs them all)."""
        raise NotImplementedError

    def text(self, id: ChunkId) -> ChunkText:
        """Returns the text of a chunk as put in the context: its header, a blank line, then its text.

        Raises KeyError for an unknown id."""
        raise NotImplementedError


def main() -> None:
    """Prints, for a question, the detected entities, the definitions, the bonus and the chunks of the context.

    Loads the index (index.load_index, index.build_store), the embedder named in its manifest, the glossary and
    the Qwen3 tokenizer. Fusion rule from the command line, MassFloor(alpha=0.5, tau=0.05) by default."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
