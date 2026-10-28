# Judge: does a matched span refer to a BOTC entity?

Version 1 — used for the threshold calibration of `botc.experiments.threshold` (§4 of the retrieval plan).

## Context

Blood on the Clocktower (BOTC) is a social deduction game. Its characters are called **roles** (Imp, Librarian,
Fortune Teller...), and its rules use **glossary terms** with a precise meaning (Player, Night, Good, Poisoned,
Execution...). A lexical matcher looks for roles and terms in sentences of the BOTC wiki by comparing character
n-grams: it finds exact names, but also look-alike words. You judge each match.

## Input

A JSONL file, one candidate per line:

- `candidate_id`: the id to copy in your answer;
- `sentence`: a sentence of the wiki, as written;
- `span`: the words of the sentence that matched, normalized (lower case, punctuation removed, "'s" removed);
- `entity`: the role or glossary term it matched;
- `description`: what the entity is — for a role, its type and ability; for a term, its glossary definition.

## Question

In this sentence, do the words of `span` refer to `entity`, in the BOTC sense given by `description`?

## Rules

1. **Role**: `true` only if the sentence talks about that character (the character, its token, its ability, or the
   player who has it). A word that only resembles the name is `false`.
   - "the Undertaker learns which character died" → Undertaker: `true`.
   - "Widow / Damsel: if the Widow is in play, the Damsel is poisoned" → Damsel: `true`.
   - "the Damsel is poisoned" → Poisoner: `false` (a state, not the Poisoner character).
   - "players claiming to be Townsfolk" → Town Crier: `false` (another word).
   - "help evil players to bluff" → Evil Twin, span "evil": `false` (the alignment, not the Evil Twin).
   - "players to bluff" → Tor, span "to": `false`.
2. **Glossary term**: `true` if the words are used with the meaning of the definition, even when they are everyday
   words. `false` if they are used in another sense.
   - "chooses an alive good player" → Player: `true`; → Good: `true` (the alignment).
   - "a good idea" → Good: `false` (not the alignment).
   - "If the Psychopath is executed, this still counts as the one execution for the day" → Execution: `true`.
   - "the Damsel is poisoned" → Poisoned: `true`.
   - "is drunk" → term Drunk: `true`; → role Drunk: `false` unless the sentence is about the Drunk character.
   - "a dead player" → Dead: `true`.
3. **Part of a name or another entity**: `false` when the span is only a fragment of a longer name used in another
   sense, or when it designates a different entity with a similar name.
   - "at the beginning of the game" → Once per game, span "game": `false`.
   - "it is unlikely that the Zombuul is among the living" → Demon, The, span "the": `false`.
   - "This will also cast doubt" → Win, span "will": `false`.
4. Judge each candidate on its own. If you hesitate after reading the description, answer `false`.

## Output

Write exactly one JSON line per input candidate, in any order, and nothing else:

```
{"candidate_id": 12, "correct": true}
{"candidate_id": 13, "correct": false}
```

Every `candidate_id` of the input must appear exactly once; `correct` is a JSON boolean (`true` / `false`).
