# dreamcatcher

Turn a verifiable interaction ledger into training data. The web filters the night: good dreams pass through to the sleeper, bad ones catch and burn at dawn.

Extracted from Bad Apple's nightly dream loop — she curates her own hash-chained ledger, trains a bounded LoRA, and gates adoption on held-out regression. **Your organism's verifiable experience becomes its next weights.**

## What the web catches

- **Pairs** — each `query` prompt held until the next `response` text; orphans drop.
- **Filters control-plane text** — tiers like `approval`, `gate_echo`, `deterministic`, `self_audit` are skipped. Training on them teaches the adapter to imitate the approval gate.
- **Filters noise** — refusals (`I'm sorry`), firewall output, gate artifacts, errors, and exchanges under 20 chars.
- **Dedupes** — exact-duplicate questions keep only the first.
- **Caps** — most recent `max_rows` pairs.
- **Splits** — last ~10% reserved as `valid` (held-out).
- **Formats** — ChatML `{"text": ...}` JSONL rows for mlx-lm-style trainers.

## The adoption gate

```rust
use dreamcatcher::{validation_losses, evaluate_adoption, Adoption};

let losses = validation_losses(&trainer_output);   // every "validation loss X" report, in order
match evaluate_adoption(&losses, 0.05) {           // 5% regression tolerance
    Adoption::Adopt { .. } => graft_the_adapter(),
    Adoption::Reject { reason } => keep_the_old_weights(&reason),
}
```

Baseline (iteration-0) vs final held-out loss. Missing eval report = reject — a trainer that can't report eval can't be adopted. This is the safety story every self-improvement proposal lacks.

## Usage

```rust
use dreamcatcher::{curate, Config};

let ledger_lines = std::fs::read_to_string("ledger.jsonl")?
    .lines().map(String::from).collect::<Vec<_>>();
let c = curate(ledger_lines.into_iter(), &Config::default());
std::fs::write("train.jsonl", c.train_jsonl())?;
std::fs::write("valid.jsonl", c.valid_jsonl())?;
```

## Why it matters

Continual learning without a filter ingests the system's own control noise — the adapter learns to refuse, apologize, and echo gates. The dreamcatcher keeps the signal and burns the noise, and the adoption gate means a bad dream never becomes weights.

## License

MIT
