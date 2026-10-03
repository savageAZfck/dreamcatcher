# dreamcatcher specification

Ledger→training-data curation and eval-gated adoption.

## Curation pipeline (`curate`)

1. **Pair.** Hold each `query` entry's `data.prompt` until the next
   `response` entry's `data.text`; orphan responses and unanswered
   queries drop.
2. **Tier filter.** Responses whose `data.tier` is in
   `untrainable_tiers` skip — control-plane text (approvals, gate
   echoes, deterministic/meta/human outputs) trains the adapter to
   imitate the gate.
3. **Content filter.** Question or answer matching `skippable` —
   under `min_text_len` or starting with a refusal/firewall/error
   prefix — skips.
4. **Dedupe.** First occurrence of each exact question wins.
5. **Cap.** Keep the most recent `max_rows` pairs.
6. **Split.** `ceil(len × valid_fraction)` (min 1) most recent pairs
   → `valid`; the rest → `train`.
7. **Format.** `chatml_row` produces the `{"text": "..."}` ChatML JSONL
   shape trainers consume.

## Adoption gate (`evaluate_adoption`)

- Input: `validation_losses(trainer_output)` — every `validation loss
  X` report in order (iteration-0 baseline first).
- Fewer than 2 losses → **reject** (no post-train eval happened).
- `final ≤ baseline × (1 + max_regression)` → **adopt**.
- Otherwise → **reject** with the regression in the reason.

## Invariants

- Control-plane text never reaches the training set.
- Held-out rows are always the *most recent* pairs — the eval measures
  the present, not the past.
- No eval report, no adoption. The gate fails closed.

## Non-goals

- Training itself — dreamcatcher produces the corpus and the gate
  verdict; the trainer is external.
- Ledger integrity — pair with `sovereign_ledger` to verify the chain
  before trusting its contents as training data.
