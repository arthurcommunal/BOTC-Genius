"""Token budget of the context: count tokens with the tokenizer of the generator, fill the context greedily."""

from collections.abc import Mapping, Sequence

from botc.retrieval.contracts import ChunkId, ModelName, TokenCount, TokenCounter

DEFAULT_GENERATOR = ModelName("Qwen/Qwen3-8B")  # every Qwen3 model shares this tokenizer


def hf_token_counter(model: ModelName = DEFAULT_GENERATOR) -> TokenCounter:
    """Returns a counter of tokens using the tokenizer of a Hugging Face model, without special tokens.

    Downloads the tokenizer files (not the weights) on first use, then reads them from the local cache."""
    raise NotImplementedError


def fill_context(ranked: Sequence[ChunkId], costs: Mapping[ChunkId, TokenCount], budget: TokenCount) -> list[ChunkId]:
    """Returns the chunks to put in the context: the longest prefix of `ranked` whose total cost is <= budget.

    The chunks are taken in rank order and the fill stops at the first one that does not fit, even if a later,
    smaller one would: the context is always the top of the ranking.
    Raises ValueError, before filling, if a chunk of `ranked` has no cost or appears twice, if a cost is negative,
    or if the budget is negative. `costs` may hold chunks absent from `ranked`.
    Hypothèse : each text is counted alone; separators and the prompt template of the generator are not counted."""
    raise NotImplementedError
