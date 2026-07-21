//! Three-way agreement: symbolic diff ≡ reverse-mode AD ≡ finite differences.
//!
//! Integration test (compiles as its own crate, using `engine` from the
//! outside). For each expression and each variable, three independently-derived
//! numbers must agree at random points:
//!   * **symbolic** — `Graph::diff` builds a derivative graph; evaluate it;
//!   * **automatic** — the reverse-mode adjoint from `Graph::backward`;
//!   * **numerical** — a central-difference estimate that only ever calls `f`.
//!
//! Symbolic and reverse-mode are both exact, so they must match tightly; the
//! finite-difference oracle shares no code with the engine and is the loose,
//! independent sanity check. Each mode compiles its own fresh graph, so the
//! nodes `diff` appends never perturb the reverse-mode pass.

use engine::parse::compile;
use rand::RngExt;
use std::collections::HashMap;

/// Central-difference step. Small enough that the O(h²) truncation error is far
/// below the assertion tolerance, large enough to avoid subtractive-cancellation
/// noise.
const H: f64 = 1e-6;

/// Forward-evaluate `src` at `point`, returning the output value. Recompiles
/// each call so the numerical oracle stays a pure black box over `f`.
fn forward_value(src: &str, point: &HashMap<String, f64>) -> f64 {
    let (mut g, root) = compile(src).expect("compile");
    g.forward(point).expect("forward");
    g.nodes[root].value
}

/// Numerical gradient by central differences, one variable at a time.
fn numerical_gradient(src: &str, point: &HashMap<String, f64>) -> HashMap<String, f64> {
    let mut grad = HashMap::new();
    let mut modified = point.clone();
    for key in point.keys() {
        let base = point[key];
        modified.insert(key.clone(), base + H);
        let f_plus = forward_value(src, &modified);
        modified.insert(key.clone(), base - H);
        let f_minus = forward_value(src, &modified);
        modified.insert(key.clone(), base); // restore
        grad.insert(key.clone(), (f_plus - f_minus) / (2.0 * H));
    }
    grad
}

/// Symbolic derivative value: build the derivative graph for `wrt`, evaluate it,
/// read the derivative root's value (not forward's last-node return).
fn symbolic_deriv(src: &str, wrt: &str, point: &HashMap<String, f64>) -> f64 {
    let (mut g, root) = compile(src).expect("compile");
    let d = g.diff(root, wrt);
    g.forward(point).expect("forward");
    g.nodes[d].value
}

/// Reverse-mode gradient: the whole gradient in one backward pass.
fn reverse_grad(src: &str, point: &HashMap<String, f64>) -> HashMap<String, f64> {
    let (mut g, root) = compile(src).expect("compile");
    g.forward(point).expect("forward");
    g.backward(root).expect("backward")
}

#[test]
fn three_way_agreement() {
    // Seven expressions (≥ 6), spanning product/quotient/power/chain and up to
    // three variables. Positive random points keep ln/div/pow in-domain.
    let cases: [(&str, &[&str]); 7] = [
        ("x * y + sin(x)", &["x", "y"]),
        ("x^2 + 3*x*y + y^2", &["x", "y"]),
        ("sin(x * y) * cos(x)", &["x", "y"]),
        ("exp(x) / (y + 1)", &["x", "y"]),
        ("ln(x) + x^3", &["x"]),
        ("(x + y) * (x - y)", &["x", "y"]),
        ("x * y * z + sin(z)", &["x", "y", "z"]),
    ];

    let mut rng = rand::rng();
    for (src, vars) in cases {
        for _ in 0..5 {
            let point: HashMap<String, f64> = vars
                .iter()
                .map(|v| (v.to_string(), rng.random_range(0.5..3.0)))
                .collect();

            let auto = reverse_grad(src, &point);
            let num = numerical_gradient(src, &point);

            for &v in vars {
                let sym = symbolic_deriv(src, v, &point);

                // Both exact: symbolic and reverse-mode must match tightly.
                assert!(
                    (sym - auto[v]).abs() < 1e-9,
                    "{src}: symbolic {sym} vs reverse {} disagree on d/d{v}",
                    auto[v]
                );
                // Independent oracle: agree to finite-difference tolerance.
                assert!(
                    (auto[v] - num[v]).abs() < 1e-4,
                    "{src}: reverse {} vs numerical {} disagree on d/d{v}",
                    auto[v],
                    num[v]
                );
            }
        }
    }
}
