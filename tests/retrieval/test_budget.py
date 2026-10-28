import pytest
from hypothesis import given
from hypothesis import strategies as st

from botc.retrieval.budget import DEFAULT_GENERATOR, fill_context, hf_token_counter
from botc.retrieval.contracts import ChunkId, TokenCount, TokenCounter

# An English paragraph of the wiki (ability and summary of the Imp).
IMP = ('"Each night*, choose a player: they die. If you kill yourself this way, a Minion becomes the Imp." '
       "The Imp kills a player each night, and can make copies of itself... for a terrible price.")


def chunks(**costs: int) -> tuple[list[ChunkId], dict[ChunkId, TokenCount]]:
    """Ranked ids in the order given, and their costs."""
    return [ChunkId(id) for id in costs], {ChunkId(id): TokenCount(cost) for id, cost in costs.items()}


def ranked_costs(costs: list[int]) -> tuple[list[ChunkId], dict[ChunkId, TokenCount]]:
    ranked = [ChunkId(f"c{i}") for i in range(len(costs))]
    return ranked, {id: TokenCount(cost) for id, cost in zip(ranked, costs)}


COSTS = st.lists(st.integers(min_value=0, max_value=300), max_size=30)
BUDGETS = st.integers(min_value=0, max_value=2000)


# --- fill_context: characterizing property ---

@given(costs=COSTS, budget=BUDGETS)
def test_the_context_is_the_longest_prefix_of_the_ranking_within_the_budget(costs: list[int], budget: int):
    """The result is a prefix of `ranked` whose total cost is <= budget, and adding the next chunk would exceed it.

    These two conditions define a single result, so checking them checks the whole function."""
    ranked, cost = ranked_costs(costs)
    taken = fill_context(ranked, cost, TokenCount(budget))
    assert taken == ranked[:len(taken)]
    used = sum(cost[id] for id in taken)
    assert used <= budget
    if len(taken) < len(ranked):
        assert used + cost[ranked[len(taken)]] > budget


# --- fill_context: examples ---

def test_the_fill_stops_at_the_first_chunk_that_does_not_fit():
    """A chunk that does not fit stops the fill, even if a later, smaller one would fit.

    Costs a 5, b 8, c 3 with a budget of 10: b does not fit after a (5 left), so c is not taken either."""
    ranked, cost = chunks(a=5, b=8, c=3)
    assert fill_context(ranked, cost, TokenCount(10)) == ["a"]


def test_the_ranking_wins_over_the_packing():
    """The best chunk is taken even if two lower ones would fill the budget better.

    Costs a 6, b 5, c 5 with a budget of 10: [a] (6 tokens), not [b, c] (10 tokens)."""
    ranked, cost = chunks(a=6, b=5, c=5)
    assert fill_context(ranked, cost, TokenCount(10)) == ["a"]


def test_a_chunk_filling_exactly_the_rest_of_the_budget_is_taken():
    """The budget is inclusive: costs a 4, b 6 fill a budget of 10 exactly."""
    ranked, cost = chunks(a=4, b=6)
    assert fill_context(ranked, cost, TokenCount(10)) == ["a", "b"]


def test_a_budget_of_zero_takes_no_chunk():
    """With nothing left (e.g. the glossary definitions used the whole budget), no chunk enters the context."""
    ranked, cost = chunks(a=1, b=2)
    assert fill_context(ranked, cost, TokenCount(0)) == []


def test_costs_of_chunks_not_ranked_are_ignored():
    """Costs are computed once for the whole index: costs of ids absent from `ranked` play no role."""
    ranked, cost = chunks(a=4, b=6)
    cost[ChunkId("zz")] = TokenCount(1)
    assert fill_context(ranked, cost, TokenCount(10)) == ["a", "b"]


# --- fill_context: errors ---

def test_a_ranked_chunk_without_cost_raises_even_after_the_stop():
    """Every ranked chunk must have a cost, checked before filling: the error does not depend on the budget.

    Here the chunk without cost comes after a chunk that already uses the whole budget."""
    ranked, cost = chunks(a=10)
    with pytest.raises(ValueError):
        fill_context([*ranked, ChunkId("zz")], cost, TokenCount(10))


@pytest.mark.parametrize("ranked, cost, budget", [
    (["a", "a"], {"a": 1}, 10),
    (["a"], {"a": -1}, 10),
    (["a"], {"a": 1}, -1),
], ids=["duplicate id", "negative cost", "negative budget"])
def test_invalid_inputs_raise(ranked: list[str], cost: dict[str, int], budget: int):
    """A ranking with a duplicate id, a negative cost or a negative budget means a bug upstream: ValueError."""
    with pytest.raises(ValueError):
        fill_context([ChunkId(id) for id in ranked], {ChunkId(id): TokenCount(c) for id, c in cost.items()},
                     TokenCount(budget))


# --- hf_token_counter (downloads the tokenizer files on first run) ---

@pytest.fixture(scope="module")
def count() -> TokenCounter:
    return hf_token_counter()


@pytest.mark.slow
def test_the_qwen3_tokenizer_counts_hello_world_as_two_tokens(count: TokenCounter):
    """Known value of the Qwen3 tokenizer: "Hello" and " world" are one token each."""
    assert count("Hello world") == 2


@pytest.mark.slow
def test_no_special_token_is_counted(count: TokenCounter):
    """The counter measures the text only: an empty text costs 0 (no beginning or end token)."""
    assert count("") == 0


@pytest.mark.slow
def test_the_counter_agrees_with_the_hugging_face_tokenizer(count: TokenCounter):
    """On a paragraph of the wiki, the count is the length of the Hugging Face encoding without special tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_GENERATOR)
    assert count(IMP) == len(tokenizer.encode(IMP, add_special_tokens=False))


@pytest.mark.slow
def test_english_text_costs_between_one_and_two_tokens_per_word(count: TokenCounter):
    """Plausibility of the unit: plain English costs 1 to 2 tokens per word, not characters nor bytes."""
    assert len(IMP.split()) <= count(IMP) <= 2 * len(IMP.split())
