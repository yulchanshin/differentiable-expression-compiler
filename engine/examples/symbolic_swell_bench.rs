//! Expression-swell benchmark for symbolic differentiation (TICKET-450).
//!
//! Symbolic `diff` produces a derivative *graph* that is conspicuously larger
//! than the original: the naive product/chain rules duplicate subtrees and
//! spray `*1`/`+0`/constant litter. Differentiate a second time and it balloons
//! again (swell squared). This benchmark measures that, then runs the Phase 4
//! optimizer (`const_fold -> cse -> dce`) on the first derivative and records how
//! far it claws the node count back.
//!
//! Counts are **reachable nodes from a root**, since after `diff` the arena
//! holds the original expression *and* the derivative; `nodes.len()` alone would
//! mix them. The optimized figure is `nodes.len()` after DCE, which compacts the
//! arena down to exactly the derivative's minimal shared form.
//!
//! Writes `bench/results/symbolic_swell.json` and prints a table. Run with
//! `cargo run --example symbolic_swell_bench` from the `engine/` crate.

use std::path::Path;

use engine::graph::arena::Graph;
use engine::parse::compile;
use serde::Serialize;

#[derive(Serialize)]
struct SwellResult {
    name: String,
    expr: String,
    wrt: String,
    original_nodes: usize,
    raw_first_deriv_nodes: usize,
    raw_second_deriv_nodes: usize,
    optimized_first_deriv_nodes: usize,
    reduction_pct: f64,
    raw_first_deriv_formula: String,
}

/// Count nodes reachable from `root` by walking input edges backward.
fn reachable(g: &Graph, root: usize) -> usize {
    let mut seen = vec![false; g.nodes.len()];
    let mut stack = vec![root];
    while let Some(i) = stack.pop() {
        if seen[i] {
            continue;
        }
        seen[i] = true;
        for &j in &g.nodes[i].inputs {
            stack.push(j);
        }
    }
    seen.iter().filter(|&&b| b).count()
}

fn measure(name: &str, src: &str, wrt: &str) -> SwellResult {
    // Original expression size.
    let (g0, root0) = compile(src).expect("compile");
    let original_nodes = reachable(&g0, root0);

    // Raw first derivative: size and formula, straight out of `diff`.
    let (mut g1, root1) = compile(src).expect("compile");
    let d1 = g1.diff(root1, wrt);
    let raw_first_deriv_nodes = reachable(&g1, d1);
    let raw_first_deriv_formula = g1.to_expr_string(d1);

    // Raw second derivative: differentiate the (unsimplified) first derivative
    // again to show the swell compounding.
    let (mut g2, root2) = compile(src).expect("compile");
    let e1 = g2.diff(root2, wrt);
    let e2 = g2.diff(e1, wrt);
    let raw_second_deriv_nodes = reachable(&g2, e2);

    // Optimized first derivative: fold constants, re-share duplicates, then DCE
    // compacts the arena to just the derivative's minimal shared form.
    let (mut g3, root3) = compile(src).expect("compile");
    g3.diff(root3, wrt);
    g3.const_fold();
    g3.cse();
    g3.dce();
    let optimized_first_deriv_nodes = g3.nodes.len();

    let reduction_pct = if raw_first_deriv_nodes == 0 {
        0.0
    } else {
        (raw_first_deriv_nodes - optimized_first_deriv_nodes) as f64 / raw_first_deriv_nodes as f64
            * 100.0
    };

    SwellResult {
        name: name.to_string(),
        expr: src.to_string(),
        wrt: wrt.to_string(),
        original_nodes,
        raw_first_deriv_nodes,
        raw_second_deriv_nodes,
        optimized_first_deriv_nodes,
        reduction_pct,
        raw_first_deriv_formula,
    }
}

fn main() {
    let results = [
        measure("nested_product", "x * y * z", "x"),
        measure("product_of_chains", "sin(x * y) * cos(x * z)", "x"),
        measure("triple_product", "(x + y) * (x - y) * (x + 2*y)", "x"),
        measure("exp_ln_mix", "exp(x * y) + ln(x + y)", "x"),
        measure("poly", "x^2 * y^2 + x * y", "x"),
    ];

    println!(
        "{:<20} {:>8} {:>10} {:>11} {:>12} {:>11}",
        "case", "original", "d/dx raw", "d²/dx² raw", "d/dx opt", "reduction"
    );
    for r in &results {
        println!(
            "{:<20} {:>8} {:>10} {:>11} {:>12} {:>10.1}%",
            r.name,
            r.original_nodes,
            r.raw_first_deriv_nodes,
            r.raw_second_deriv_nodes,
            r.optimized_first_deriv_nodes,
            r.reduction_pct
        );
    }
    let raw_total: usize = results.iter().map(|r| r.raw_first_deriv_nodes).sum();
    let opt_total: usize = results.iter().map(|r| r.optimized_first_deriv_nodes).sum();
    let overall = (raw_total - opt_total) as f64 / raw_total as f64 * 100.0;
    println!(
        "{:<20} {:>8} {:>10} {:>11} {:>12} {:>10.1}%",
        "TOTAL", "", raw_total, "", opt_total, overall
    );

    // Show one derivative rendered as a formula to make the swell concrete.
    println!(
        "\nraw d/dx of {:?}:\n  {}",
        results[0].expr, results[0].raw_first_deriv_formula
    );

    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine/ has a parent")
        .join("bench/results");
    std::fs::create_dir_all(&out_dir).expect("create bench/results");
    let out_path = out_dir.join("symbolic_swell.json");
    let json = serde_json::to_string_pretty(&results).expect("serialize results");
    std::fs::write(&out_path, json + "\n").expect("write symbolic_swell.json");
    println!("\nwrote {}", out_path.display());
}
