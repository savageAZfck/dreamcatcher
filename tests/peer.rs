#![cfg(feature = "peer-signing")]

use dreamcatcher::{chatml_row, curate, merge_peer_rows, Config, Curated, DreamShare, Stats};
use ed25519_dalek::{Signer, SigningKey};

fn key(byte: u8) -> SigningKey {
    SigningKey::from_bytes(&[byte; 32])
}

fn corpus() -> Curated {
    corpus_of(2)
}

fn corpus_of(n: usize) -> Curated {
    let mut lines = Vec::new();
    for i in 0..n {
        lines.push(format!(
            r#"{{"type":"query","data":{{"prompt":"question {i} about the world"}}}}"#
        ));
        lines.push(format!(
            r#"{{"type":"response","data":{{"text":"answer {i} long enough to survive the filter and matter"}}}}"#
        ));
    }
    curate(lines.into_iter(), &Config::default())
}

#[test]
fn emit_and_verify() {
    let c = corpus();
    let share = DreamShare::emit(&key(1), &c, 8);
    assert_eq!(share.rows, c.train.len());
    assert_eq!(share.corpus_digest.len(), 64);
    assert!(share.verify());
}

#[test]
fn tampered_lesson_fails_verify() {
    let c = corpus();
    let mut share = DreamShare::emit(&key(1), &c, 8);
    share.lessons.push(chatml_row("injected", "poisoned"));
    assert!(!share.verify());
}

#[test]
fn foreign_signature_fails() {
    let c = corpus();
    let mut share = DreamShare::emit(&key(1), &c, 8);
    // Re-sign with a different key but keep the original node id.
    share.signature = hex::encode(
        key(9)
            .sign(serde_json::to_string(&share.payload()).unwrap().as_bytes())
            .to_bytes(),
    );
    assert!(!share.verify());
}

#[test]
fn merge_respects_cap_and_dedupes() {
    let mut local = corpus_of(20);
    let local_n = local.train.len();
    let mut peer_corpus = Curated {
        train: vec![],
        valid: vec![],
        stats: Stats::default(),
    };
    for i in 0..50 {
        peer_corpus.train.push(chatml_row(
            &format!("peer lesson {i}"),
            "an answer long enough to matter at all",
        ));
    }
    let share = DreamShare::emit(&key(2), &peer_corpus, 50);
    let merged = merge_peer_rows(&mut local, &[share], 0.25);
    // cap = local * 0.25 / 0.75 → strictly bounded, never peers-majority
    assert!(merged > 0 && local.train.len() > local_n);
    assert!(merged as f64 <= local_n as f64 * 0.5);
}
