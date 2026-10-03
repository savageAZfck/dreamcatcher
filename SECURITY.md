# Security policy

## Reporting

Open a private security advisory on the GitHub repository, or email the
maintainer (see `Cargo.toml` authors). Do not file public issues for
data-poisoning vectors — e.g. a ledger entry shape that bypasses the
control-plane filter.

## Scope

In scope:

- Filter bypasses that let control-plane tiers or refusal content
  reach the training corpus.
- `evaluate_adoption` returning `Adopt` on missing or regressed eval.
- Loss-extraction parsing that fabricates a baseline/final from
  unrelated output.
- Pairing bugs that attribute a response to the wrong question.

Out of scope:

- Malicious-but-plausible training pairs — semantic poisoning is a
  ledger-integrity problem (`sovereign_ledger`, ingestion gating).
- Trainer-reported numbers — taken at face value inside the trust
  boundary.
- The trained adapter's downstream behavior — adoption is gated on
  loss, not alignment; pair with a behavioral eval.

## Guidance

- Verify the source ledger's chain before curating — poisoned history
  makes poisoned weights.
- Keep `max_regression` tight (≤0.05); the gate is only as strong as
  the tolerance you set.
- Anchor adoption decisions in a constitutional record (`xipe`) so a
  graft is never unilateral.
