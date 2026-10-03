use dreamcatcher::{curate, Config};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let mut lines = Vec::new();
    for i in 0..10_000 {
        lines.push(format!(
            r#"{{"type":"query","data":{{"prompt":"question {i} that is certainly long enough"}}}}"#
        ));
        lines.push(format!(
            r#"{{"type":"response","data":{{"text":"a thorough and genuine answer to question {i} with length","tier":"model"}}}}"#
        ));
    }
    let t = Instant::now();
    let c = curate(black_box(lines.into_iter()), black_box(&Config::default()));
    println!(
        "curate 10k pairs → {} train/{} valid in {:?}",
        c.stats.train,
        c.stats.valid,
        t.elapsed()
    );
}
