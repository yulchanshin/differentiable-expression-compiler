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
