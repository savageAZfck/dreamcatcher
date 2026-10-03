use dreamcatcher::{chatml_row, curate, evaluate_adoption, validation_losses, Adoption, Config};

fn ledger(pairs: &[(&str, &str, Option<&str>)]) -> Vec<String> {
    let mut lines = Vec::new();
    for (q, a, tier) in pairs {
        lines.push(format!(
            r#"{{"type":"query","data":{{"prompt":"{q}"}}}}"#
        ));
        let tier_field = tier
            .map(|t| format!(r#","tier":"{t}""#))
            .unwrap_or_default();
        lines.push(format!(
            r#"{{"type":"response","data":{{"text":"{a}"{tier_field}}}}}"#
        ));
    }
    lines
}

const LONG_ANSWER: &str = "This is a real answer with enough length to pass the filter.";

#[test]
fn pairs_query_with_next_response() {
    let l = ledger(&[("what is a hash-chained ledger?", LONG_ANSWER, None)]);
    let c = curate(l.into_iter(), &Config::default());
    assert_eq!(c.train.len() + c.valid.len(), 1);
    assert_eq!(c.stats.candidates, 1);
}

#[test]
fn control_plane_tiers_are_filtered() {
    let l = ledger(&[
        ("question one that is long enough", LONG_ANSWER, Some("approval")),
        ("question two that is long enough", LONG_ANSWER, Some("deterministic")),
        ("question three that is long enough", LONG_ANSWER, Some("gate_echo")),
        ("question four that is long enough", LONG_ANSWER, Some("model")),
    ]);
    let c = curate(l.into_iter(), &Config::default());
    // Only q4 survives — training on gate echoes teaches the adapter
    // to imitate the approval gate.
    assert_eq!(c.stats.skipped, 3);
    assert_eq!(c.train.len() + c.valid.len(), 1);
}

#[test]
fn refusals_and_short_text_are_noise() {
    let l = ledger(&[
        ("question one that is long enough", "I'm sorry, I can't do that.", None),
        ("question two that is long enough", "ok", None),
        ("question three that is long enough", "[Output firewall: blocked]", None),
        ("question four that is long enough", LONG_ANSWER, None),
    ]);
    let c = curate(l.into_iter(), &Config::default());
    assert_eq!(c.stats.skipped, 3);
    assert_eq!(c.train.len() + c.valid.len(), 1);
}

#[test]
fn dedupes_questions() {
    let l = ledger(&[
        ("same question asked twice now", LONG_ANSWER, None),
        ("same question asked twice now", LONG_ANSWER, None),
        ("a different question entirely", LONG_ANSWER, None),
    ]);
    let c = curate(l.into_iter(), &Config::default());
    assert_eq!(c.stats.deduped, 1);
    assert_eq!(c.train.len() + c.valid.len(), 2);
}

#[test]
fn splits_last_ten_percent_to_valid() {
    let pairs: Vec<_> = (0..20)
        .map(|i| {
            (
                Box::leak(format!("question number {i} long enough").into_boxed_str()) as &str,
                LONG_ANSWER,
                None,
            )
        })
        .collect();
    let l = ledger(&pairs);
    let c = curate(l.into_iter(), &Config::default());
    assert_eq!(c.stats.valid, 2);
    assert_eq!(c.stats.train, 18);
}

#[test]
fn chatml_row_format() {
    let row = chatml_row("q?", "a!");
    assert!(row.contains("im_start"));
    assert!(row.contains("user"));
    assert!(row.contains("assistant"));
    let v: serde_json::Value = serde_json::from_str(&row).unwrap();
    assert!(v["text"].as_str().unwrap().contains("q?"));
}

#[test]
fn validation_loss_extraction() {
    let out = "Iter 1: validation loss 2.345\nIter 40: validation loss 1.987e-1\nIter 80: validation loss 1.50";
    assert_eq!(validation_losses(out), vec![2.345, 0.1987, 1.50]);
}

#[test]
fn adoption_gate() {
    // Improvement → adopt.
    assert!(matches!(
        evaluate_adoption(&[2.0, 1.5], 0.05),
        Adoption::Adopt { .. }
    ));
    // Within tolerance → adopt.
    assert!(matches!(
        evaluate_adoption(&[2.0, 2.05], 0.05),
        Adoption::Adopt { .. }
    ));
    // Regression beyond tolerance → reject.
    assert!(matches!(
        evaluate_adoption(&[2.0, 2.2], 0.05),
        Adoption::Reject { .. }
    ));
    // Missing eval → reject (a trainer that can't report can't be adopted).
    assert!(matches!(
        evaluate_adoption(&[2.0], 0.05),
        Adoption::Reject { .. }
    ));
    assert!(matches!(
        evaluate_adoption(&[], 0.05),
        Adoption::Reject { .. }
    ));
}
