"""Token budget of the context: count tokens with the tokenizer of the generator, fill the context with the top
of the ranking."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from botc.retrieval.contracts import ChunkId, ModelName, TokenCount, TokenCounter

DEFAULT_GENERATOR = ModelName("Qwen/Qwen3-8B")  # every Qwen3 model shares this tokenizer


def hf_token_counter(model: ModelName = DEFAULT_GENERATOR) -> TokenCounter:
    """Returns a counter of tokens using the tokenizer of a Hugging Face model, without special tokens.

    Downloads the tokenizer files (not the weights) on first use, then reads them from the local cache."""
    # Heavy import, only when a real counter is needed. Its types are partly unknown: the tokenizer is used through Any.
    from transformers import AutoTokenizer

    tokenizer = cast(Any, AutoTokenizer.from_pretrained(model))  # pyright: ignore[reportUnknownMemberType]

    def count(text: str) -> TokenCount:
        return TokenCount(len(tokenizer.encode(text, add_special_tokens=False)))

    return count


def fill_context(ranked: Sequence[ChunkId], costs: Mapping[ChunkId, TokenCount], budget: TokenCount) -> list[ChunkId]:
    """Returns the chunks to put in the context: the longest prefix of `ranked` whose total cost is <= budget.

    The chunks are taken in rank order and the fill stops at the first one that does not fit, even if a later,
    smaller one would: the context is always the top of the ranking.
    Raises ValueError, before filling, if a chunk of `ranked` has no cost or appears twice, if a cost is negative,
    or if the budget is negative. `costs` may hold chunks absent from `ranked`.
    Hypothèse : each text is counted alone; separators and the prompt template of the generator are not counted."""
    if budget < 0:
        raise ValueError(f"Negative budget: {budget}")
    if len(set(ranked)) != len(ranked):
        raise ValueError("Duplicate chunk ids in the ranking")
    missing = [id for id in ranked if id not in costs]
    if missing:
        raise ValueError(f"{len(missing)} ranked chunks have no cost, first ones: {missing[:5]}")
    negative = [id for id in ranked if costs[id] < 0]
    if negative:
        raise ValueError(f"Negative costs for {negative[:5]}")
    taken: list[ChunkId] = []
    left = budget
    for id in ranked:
        if costs[id] > left:
            break
        taken.append(id)
        left -= costs[id]
    return taken
