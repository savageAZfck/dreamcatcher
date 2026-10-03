# Threat model

dreamcatcher assumes the ledger is honest history and the trainer
reports losses truthfully — it is the *curation and gate* layer between
raw experience and new weights.

## What it defends against

- **Control-plane contamination.** Gate prompts, approval echoes, and
  deterministic outputs are filtered — an adapter trained on them
  learns to imitate the brake, not the mind.
- **Refusal/noise poisoning.** Refusals, firewall output, errors, and
  trivially short exchanges never reach the corpus.
- **Adoption without evidence.** The eval gate fails closed: missing
  validation reports reject outright; >`max_regression` held-out
  regression rejects. A bad dream never becomes weights.
- **Recency bias in eval.** Held-out rows are the most recent pairs —
  the gate measures current behavior, not stale history.

## What it does not defend against

- **A poisoned ledger.** If an attacker can inject query/response
  pairs into the ledger, they can inject training data. Verify the
  ledger's hash chain (`sovereign_ledger`) and gate ingestion upstream.
- **Subtle semantic poisoning.** Statistically valid, semantically
  malicious pairs pass the filters — the gate catches *loss*
  regression, not *intent* regression. Behavioral A/B eval after
  adoption is the complement.
- **Trainer dishonesty.** Reported losses are taken at face value;
  a trainer that lies about eval defeats the gate. Keep the trainer
  in the trust boundary.

## Design posture

Filter the night honestly, hold out the present for eval, and never
let a dream become weights without beating its own baseline.
