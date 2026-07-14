//! Golden-file trace test (TICKET-205).
//!
//! Integration test: compiles as its own crate and drives `engine` from the
//! outside via `use engine::...`, the same shape as the finite-difference
//! oracle in `finite_diff.rs`.
//!
//! Pins the serialized §7.3 trace for `sin(x*y) + x^2` at (x, y) = (1.5, 2.0)
//! against a checked-in golden file, and asserts the two ordering invariants
//! the contract guarantees: `forward` covers every node in topological order,
//! and `backward` is exactly its reverse.
//!
//! After an *intentional* format change, regenerate the golden with:
//!   UPDATE_GOLDEN=1 cargo test --test trace_golden

use engine::graph::arena::Graph;
use std::collections::HashMap;
use std::path::PathBuf;

/// Build `sin(x*y) + x^2`, matching §7.3's node numbering (ids 0..=5).
///
/// The arena pushes nodes in construction order and every op is created after
/// its inputs, so a node's index is always greater than its inputs' — index
/// order is therefore a valid topological order, which is what `trace` emits.
fn build_sin_xy_plus_x2() -> Graph {
    let mut g = Graph::new();
    let x = g.var("x".into()); // 0
    let y = g.var("y".into()); // 1
    let xy = g.mul(x, y); // 2
    let s = g.sin(xy); // 3
    let x2 = g.pow(x, 2.0); // 4
    let _f = g.add(s, x2); // 5
    g
}

fn fixed_point() -> HashMap<String, f64> {
    HashMap::from([("x".to_string(), 1.5), ("y".to_string(), 2.0)])
}

fn golden_path() -> PathBuf {
    // CARGO_MANIFEST_DIR is the `engine/` crate root at both compile and run time.
    [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "golden",
        "trace_sin_xy_plus_x2.json",
    ]
    .iter()
    .collect()
}

// Acceptance criteria 1 + 3: `serde_json::to_string(&trace)` matches the §7.3
// shape, pinned against a checked-in golden file.
#[test]
fn trace_matches_golden_json() {
    let mut g = build_sin_xy_plus_x2();
    let output = g.nodes.len() - 1;

    let trace = g.trace(&fixed_point(), output).expect("trace should succeed");
    let mut json = serde_json::to_string_pretty(&trace).expect("serialize trace");
    json.push('\n'); // keep the golden file newline-terminated

    let path = golden_path();
    if std::env::var("UPDATE_GOLDEN").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).expect("create golden dir");
        std::fs::write(&path, &json).expect("write golden");
        return;
    }

    let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing golden file {}; regenerate with \
             UPDATE_GOLDEN=1 cargo test --test trace_golden",
            path.display()
        )
    });
    assert_eq!(
        json, expected,
        "trace JSON drifted from the golden file; if this change is intentional, \
         regenerate with UPDATE_GOLDEN=1 cargo test --test trace_golden"
    );
}

// Acceptance criterion 2: `forward.len() == node_count`, and `backward` is
// exactly `forward` reversed.
#[test]
fn forward_is_topological_and_backward_is_its_reverse() {
    let mut g = build_sin_xy_plus_x2();
    let node_count = g.nodes.len();
    let output = node_count - 1;

    let trace = g.trace(&fixed_point(), output).expect("trace should succeed");

    // `forward` visits every node once, in index (topological) order.
    assert_eq!(trace.forward.len(), node_count);
    for (i, step) in trace.forward.iter().enumerate() {
        assert_eq!(step.id, i, "forward step {i} out of topological order");
    }

    // `backward` mirrors `forward`: same length, ids in reverse.
    assert_eq!(trace.backward.len(), node_count);
    let forward_ids: Vec<usize> = trace.forward.iter().map(|s| s.id).collect();
    let backward_ids: Vec<usize> = trace.backward.iter().map(|s| s.id).collect();
    let reversed: Vec<usize> = forward_ids.iter().rev().copied().collect();
    assert_eq!(backward_ids, reversed, "backward is not the reverse of forward");
}
