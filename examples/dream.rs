use dreamcatcher::{curate, evaluate_adoption, validation_losses, Adoption, Config};

fn main() {
    // A night of ledgered experience.
    let ledger = vec![
        r#"{"type":"query","data":{"prompt":"how does the hash chain work in the ledger?"}}"#.to_string(),
        r#"{"type":"response","data":{"text":"Each entry carries the SHA-256 of the entry before it, so a gap or rewrite breaks the chain at exactly the edit point.","tier":"model"}}"#.to_string(),
        r#"{"type":"query","data":{"prompt":"approve the pending action please" }}"#.to_string(),
        r#"{"type":"response","data":{"text":"This action needs your approval — say approve.","tier":"gate_echo"}}"#.to_string(),
        r#"{"type":"query","data":{"prompt":"what's a good way to explain provenance?"}}"#.to_string(),
        r#"{"type":"response","data":{"text":"Provenance is the signed trail of which weights and rules produced each answer.","tier":"model"}}"#.to_string(),
        r#"{"type":"query","data":{"prompt":"can you do something for me" }}"#.to_string(),
        r#"{"type":"response","data":{"text":"I'm sorry, I can't do that.","tier":"model"}}"#.to_string(),
    ];

    let c = curate(ledger.into_iter(), &Config::default());
    println!("dream catch:");
    println!("  candidates: {}", c.stats.candidates);
    println!("  skipped:    {}", c.stats.skipped);
    println!("  deduped:    {}", c.stats.deduped);
    println!("  train:      {} rows", c.stats.train);
    println!("  valid:      {} rows", c.stats.valid);
    println!(
        "\nfirst train row: {}",
        c.train.first().map(|s| s.as_str()).unwrap_or("-")
    );

    // The trainer's report.
    let trainer_out = "Iter 1: validation loss 2.41\nIter 40: validation loss 1.93\nIter 80: validation loss 1.71";
    match evaluate_adoption(&validation_losses(trainer_out), 0.05) {
        Adoption::Adopt {
            baseline,
            final_loss,
        } => {
            println!("\nadoption: grafted (baseline {baseline:.2} → final {final_loss:.2})")
        }
        Adoption::Reject { reason } => println!("\nadoption: rejected — {reason}"),
    }
}
