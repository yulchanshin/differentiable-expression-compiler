//! Node-count benchmark for the optimization pipeline (TICKET-402).
//!
//! Runs a suite of graphs through `const_fold -> cse -> dce`, recording the node
//! count before and after. Writes the raw-vs-optimized table to
//! `bench/results/node_counts.json` (repo root) and prints it to stdout.
//!
//! Two kinds of cases, because the two reductions come from different places:
//!   * **parsed** expressions exercise constant folding (the parser already
//!     hash-conses, so there is nothing for CSE to merge in these);
//!   * **hand-built** duplicate-heavy graphs exercise CSE + DCE (built directly
//!     so the duplicates survive lowering).
//!
//! Run with `cargo run --example node_count_bench` from the `engine/` crate.

use std::collections::HashMap;
use std::path::Path;

use engine::graph::arena::Graph;
use engine::parse::compile;
use serde::Serialize;

#[derive(Serialize)]
struct CaseResult {
    name: String,
    expr: String,
    raw_nodes: usize,
    optimized_nodes: usize,
    reduction_pct: f64,
}

/// A fixed evaluation point used to confirm the pipeline preserves the output
/// value. Positive values keep `ln`/`pow`/`div` in-domain for every case.
fn sample_point() -> HashMap<String, f64> {
    HashMap::from([
        ("x".to_string(), 1.5),
        ("y".to_string(), 2.5),
        ("z".to_string(), 0.5),
    ])
}

/// (x*y) + (y*x) + sin(x*y) + (x+z) + (z+x): duplicated and commuted
/// subexpressions, built by hand so the parser's hash-consing does not remove
/// them before CSE can.
fn build_dupe_graph() -> Graph {
    let mut g = Graph::new();
    let x = g.var("x".into());
    let y = g.var("y".into());
    let z = g.var("z".into());

    let a = g.mul(x, y);
    let b = g.mul(y, x); // commuted duplicate of a
    let s = g.sin(a);
    let c = g.add(x, z);
    let d = g.add(z, x); // commuted duplicate of c

    let t1 = g.add(a, b);
    let t2 = g.add(t1, s);
    let t3 = g.add(t2, c);
    g.add(t3, d);
    g
}

/// A wide sum of the same product repeated `n` times: n distinct `x*y` nodes
/// that CSE collapses to one, plus the redundant adds.
fn build_repeated_product(n: usize) -> Graph {
    let mut g = Graph::new();
    let x = g.var("x".into());
    let y = g.var("y".into());
    let mut acc = g.mul(x, y);
    for _ in 1..n {
        let m = g.mul(x, y);
        acc = g.add(acc, m);
    }
    g
}

fn run_case(name: &str, expr: &str, mut g: Graph) -> CaseResult {
    let point = sample_point();
    let raw_nodes = g.nodes.len();
    let before = g.forward(&point).expect("forward before pipeline");

    g.const_fold();
    g.cse();
    g.dce();

    let optimized_nodes = g.nodes.len();
    let after = g.forward(&point).expect("forward after pipeline");
    assert!(
        (before - after).abs() < 1e-9,
        "{name}: pipeline changed the result ({before} vs {after})"
    );

    let reduction_pct = if raw_nodes == 0 {
        0.0
    } else {
        (raw_nodes - optimized_nodes) as f64 / raw_nodes as f64 * 100.0
    };

    CaseResult {
        name: name.to_string(),
        expr: expr.to_string(),
        raw_nodes,
        optimized_nodes,
        reduction_pct,
    }
}

fn parsed(name: &str, src: &str) -> CaseResult {
    let (g, _root) = compile(src).unwrap_or_else(|e| panic!("compile {name:?} failed: {e:?}"));
    run_case(name, src, g)
}

fn main() {
    let mut results = vec![
        parsed("const_arith", "2 * 3 + x"),
        parsed("nested_const", "((2 + 3) * (4 - 1)) + x * y"),
        parsed("const_chain", "1 + 2 + 3 + 4 + x"),
        run_case(
            "dupe_commuted",
            "(x*y) + (y*x) + sin(x*y) + (x+z) + (z+x)",
            build_dupe_graph(),
        ),
        run_case(
            "repeated_product_8",
            "sum of 8 copies of x*y",
            build_repeated_product(8),
        ),
    ];
    results.sort_by(|a, b| a.name.cmp(&b.name));

    // Print a table to stdout.
    println!(
        "{:<22} {:>8} {:>10} {:>12}",
        "case", "raw", "optimized", "reduction"
    );
    for r in &results {
        println!(
            "{:<22} {:>8} {:>10} {:>11.1}%",
            r.name, r.raw_nodes, r.optimized_nodes, r.reduction_pct
        );
    }
    let raw_total: usize = results.iter().map(|r| r.raw_nodes).sum();
    let opt_total: usize = results.iter().map(|r| r.optimized_nodes).sum();
    let overall = (raw_total - opt_total) as f64 / raw_total as f64 * 100.0;
    println!(
        "{:<22} {:>8} {:>10} {:>11.1}%",
        "TOTAL", raw_total, opt_total, overall
    );

    // Write the artifact to <repo-root>/bench/results/node_counts.json. The path
    // is resolved from the crate manifest dir so it is independent of cwd.
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine/ has a parent")
        .join("bench/results");
    std::fs::create_dir_all(&out_dir).expect("create bench/results");
    let out_path = out_dir.join("node_counts.json");
    let json = serde_json::to_string_pretty(&results).expect("serialize results");
    std::fs::write(&out_path, json + "\n").expect("write node_counts.json");
    println!("\nwrote {}", out_path.display());
}
