//! dreamcatcher — turn a verifiable interaction ledger into training
//! data.
//!
//! The web filters the night: good dreams pass through to the sleeper,
//! bad ones catch and burn at dawn. This crate curates an append-only
//! ledger of `query`/`response` events into `train.jsonl`/`valid.jsonl`
//! pairs — filtering out control-plane text (gate prompts, deterministic
//! outputs, refusals) so the adapter learns *genuine answers*, never
//! the sound of its own brake pedal.
//!
//! Extracted from Bad Apple's nightly dream loop, which curates her own
//! hash-chained ledger, trains a bounded LoRA, and gates adoption on
//! held-out regression.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

// ─────────────────────────── configuration ──────────────────────────

/// Curation policy — defaults ported from Bad Apple's dream pass.
#[derive(Debug, Clone)]
pub struct Config {
    /// Keep at most this many of the most recent pairs.
    pub max_rows: usize,
    /// Fraction of pairs reserved for held-out validation.
    pub valid_fraction: f64,
    /// Minimum text length — shorter exchanges are noise.
    pub min_text_len: usize,
    /// Response tiers that are control-plane, not cognition.
    pub untrainable_tiers: HashSet<String>,
    /// Content prefixes that mark a text as untrainable.
    pub skip_prefixes: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_rows: 3_000,
            valid_fraction: 0.10,
            min_text_len: 20,
            untrainable_tiers: [
                "deterministic",
                "fast",
                "meta",
                "approval",
                "gate_echo",
                "human_tool",
                "human_command",
                "human_parse_error",
                "self_audit",
                "introspection",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            skip_prefixes: [
                "I'm sorry",
                "I am sorry",
                "[Output firewall",
                "Approval required",
                "approve ",
                "deny ",
                "kill switch",
                "Unknown tool",
                "Error",
                "This action needs your approval",
                "Policy gate:",
                "Policy:",
                "I meant to use a tool",
                "[system] a gated action",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }
}

impl Config {
    /// The content filter: true when `text` is noise that must not be
    /// learned — too short, a refusal, a gate artifact, an error.
    pub fn skippable(&self, text: &str) -> bool {
        if text.len() < self.min_text_len {
            return true;
        }
        self.skip_prefixes.iter().any(|p| text.starts_with(p.as_str()))
    }
}

// ─────────────────────────── curation ───────────────────────────────

/// Curation statistics — what the web caught and what it dropped.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Stats {
    /// Query/response pairs considered.
    pub candidates: usize,
    /// Dropped for untrainable tier or content filter.
    pub skipped: usize,
    /// Dropped as exact-duplicate questions.
    pub deduped: usize,
    /// Rows written to `train`.
    pub train: usize,
    /// Rows written to `valid`.
    pub valid: usize,
}

/// The curated corpus.
#[derive(Debug, Clone)]
pub struct Curated {
    /// ChatML-formatted JSONL rows for training.
    pub train: Vec<String>,
    /// Held-out rows — adoption is gated on regression measured here.
    pub valid: Vec<String>,
    pub stats: Stats,
}

impl Curated {
    /// `train` as an NDJSON body (trailing newline).
    pub fn train_jsonl(&self) -> String {
        self.train.join("\n") + "\n"
    }
    /// `valid` as an NDJSON body (trailing newline).
    pub fn valid_jsonl(&self) -> String {
        self.valid.join("\n") + "\n"
    }
}

/// One exchange, extracted from the ledger.
#[derive(Debug, Clone)]
struct Pair {
    question: String,
    answer: String,
    tier: Option<String>,
}

/// Format a pair as a ChatML row — the `{"text": "..."}` JSONL shape
/// that mlx-lm / lora trainers consume.
pub fn chatml_row(question: &str, answer: &str) -> String {
    // Written with escapes so the tags never appear literally.
    let start = "\u{3c}\u{7c}im_start\u{7c}\u{3e}";
    let end = "\u{3c}\u{7c}im_end\u{7c}\u{3e}";
    let text = format!("{start}user\n{question}{end}\n{start}assistant\n{answer}{end}\n");
    serde_json::json!({ "text": text }).to_string()
}

// ─────────────────────── ledger → corpus ────────────────────────────

/// Curate ledger lines in Bad Apple's NDJSON format (`type` +
/// `data.{prompt|text|tier}`) into train/valid rows.
///
/// Pairing rule: each `query` entry's `prompt` is held until the next
/// `response` entry's `text` — a response without a pending question is
/// dropped, as is a question the response never answered.
pub fn curate(lines: impl Iterator<Item = String>, config: &Config) -> Curated {
    let mut pairs: Vec<Pair> = Vec::new();
    let mut pending: Option<String> = None;
    let mut stats = Stats::default();

    for line in lines {
        if !line.starts_with('{') {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(ty) = v["type"].as_str() else { continue };
        let data = &v["data"];
        match ty {
            "query" => {
                pending = data["prompt"].as_str().map(String::from);
            }
            "response" => {
                let Some(question) = pending.take() else { continue };
                let Some(text) = data["text"].as_str() else { continue };
                stats.candidates += 1;
                pairs.push(Pair {
                    question,
                    answer: text.to_string(),
                    tier: data["tier"].as_str().map(String::from),
                });
            }
            _ => continue,
        }
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut kept: Vec<Pair> = Vec::new();
    for p in pairs {
        // Control-plane text trains the adapter to imitate the gate.
        if p.tier
            .as_ref()
            .map(|t| config.untrainable_tiers.contains(t))
            .unwrap_or(false)
            || config.skippable(&p.question)
            || config.skippable(&p.answer)
        {
            stats.skipped += 1;
            continue;
        }
        if !seen.insert(p.question.clone()) {
            stats.deduped += 1;
            continue;
        }
        kept.push(p);
    }

    // Cap at the most recent rows.
    let start = kept.len().saturating_sub(config.max_rows);
    let kept = &kept[start..];
    let valid_count = ((kept.len() as f64) * config.valid_fraction).ceil().max(1.0) as usize;
    let valid_count = valid_count.min(kept.len());
    let (train, valid) = kept.split_at(kept.len() - valid_count);

    stats.train = train.len();
    stats.valid = valid.len();
    Curated {
        train: train.iter().map(|p| chatml_row(&p.question, &p.answer)).collect(),
        valid: valid.iter().map(|p| chatml_row(&p.question, &p.answer)).collect(),
        stats,
    }
}

// ─────────────────────────── adoption gate ──────────────────────────

/// Extract every `validation loss X` report from trainer output, in
/// order. The iteration-0 baseline is always emitted first.
pub fn validation_losses(output: &str) -> Vec<f64> {
    const NEEDLE: &str = "validation loss";
    let lower = output.to_lowercase();
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(off) = lower[pos..].find(NEEDLE) {
        let start = pos + off + NEEDLE.len();
        let rest = lower[start..].trim_start();
        let num_end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-'))
            .unwrap_or(rest.len());
        if let Ok(n) = rest[..num_end].parse::<f64>() {
            out.push(n);
        }
        pos = start;
    }
    out
}

/// Whether a trained candidate may be adopted.
#[derive(Debug, Clone, PartialEq)]
pub enum Adoption {
    /// Held-out loss improved or held within tolerance.
    Adopt { baseline: f64, final_loss: f64 },
    /// Missing or regressed validation — reject.
    Reject { reason: String },
}

/// Gate adoption on held-out regression: the candidate is adopted only
/// when its final validation loss is at most `(1 + max_regression)`
/// times the iteration-0 baseline. Missing losses reject — a trainer
/// that can't report eval can't be adopted.
pub fn evaluate_adoption(losses: &[f64], max_regression: f64) -> Adoption {
    let (Some(&baseline), Some(&final_loss)) = (losses.first(), losses.last()) else {
        return Adoption::Reject {
            reason: "missing validation loss reports".to_string(),
        };
    };
    if losses.len() < 2 {
        return Adoption::Reject {
            reason: "only baseline reported — no post-train eval".to_string(),
        };
    }
    let ceiling = baseline * (1.0 + max_regression);
    if final_loss <= ceiling {
        Adoption::Adopt {
            baseline,
            final_loss,
        }
    } else {
        Adoption::Reject {
            reason: format!(
                "regression: final validation loss {final_loss:.4} exceeds {ceiling:.4} \
                 (baseline {baseline:.4} + {:.0}% tolerance)",
                max_regression * 100.0
            ),
        }
    }
}