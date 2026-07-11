//! Finite-difference oracle: the AD correctness harness (TICKET-104).
//!
//! Integration test: compiles as its own crate and exercises `engine`
//! from the outside via `use engine::...`. The real oracle lands in
//! TICKET-104 once the autodiff core exists; this file just stakes out
//! the location.

use engine::graph::arena::Graph;
use std::collections::HashMap;

fn numerical_gradient(
    mut f: impl FnMut(&HashMap<String, f64>) -> f64,
    point: &HashMap<String, f64>,
    h: f64,
) -> HashMap<String, f64> {
    let mut grad: HashMap<String, f64> = HashMap::new();
    let mut modified: HashMap<String, f64> = point.clone();
    for key in point.keys() {
        let base: f64 = point[key];
        modified.insert(key.clone(), base + h);
        let f_plus: f64 = f(&modified);
        modified.insert(key.clone(), base - h);
        let f_minus: f64 = f(&modified);
        modified.insert(key.clone(), base); //restore 
        grad.insert(key.clone(), (f_plus - f_minus) / (2.0 * h));
    }
    grad
}

fn check(build: impl Fn(&mut Graph) -> usize, point: &HashMap<String, f64>, tolerance: f64) {
    //Automatic Differentiation
    let mut g = Graph::new();
    build(&mut g);
    g.forward(point);
    let auto_diff: HashMap<String, f64> = g.backward();

    //Numerical Differentiation
    let f = |inputs: &HashMap<String, f64>| {
        let mut gg = Graph::new();
        build(&mut gg);
        gg.forward(inputs)
    };

    let num_grad: HashMap<String, f64> = numerical_gradient(f, point, 1e-5);

    for key in point.keys() {
        assert!(
            (auto_diff[key] - num_grad[key]).abs() < tolerance,
            "grad mismatch on {key}: auto_diff={}, num_grad={}",
            auto_diff[key],
            num_grad[key],
        );
    }
}
