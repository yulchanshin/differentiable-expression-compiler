//! Forward evaluation over the arena.
//!
//! Fills every node's `value` by iterating the graph in topological order and
//! `match`ing on `OpType`, reading each node's inputs' already-computed values.
//! Domain failures (unknown variable, division by zero, `ln`/`pow` domain) are
//! returned as [`EngineError`] rather than producing `NaN`/`inf` or panicking.

use crate::error::EngineError;
use crate::graph::arena::Graph;
use crate::graph::node::OpType;
use crate::ops::eval::eval_op;
use std::collections::HashMap;

// Index order is a valid topological order: builder helpers always push a
// node's inputs before the node itself, so every input sits at a lower index.
impl Graph {
    pub fn forward(&mut self, inputs: &HashMap<String, f64>) -> Result<f64, EngineError> {
        for i in 0..self.nodes.len() {
            let value = match &self.nodes[i].op {
                OpType::Const(c) => *c,
                OpType::Var(name) => *inputs
                    .get(name)
                    .ok_or_else(|| EngineError::UnknownVariable(name.clone()))?,
                // Every operator node shares one scalar-eval definition with the
                // constant-folding pass so their semantics can't drift.
                _ => {
                    let vals: Vec<f64> = self.nodes[i]
                        .inputs
                        .iter()
                        .map(|&k| self.nodes[k].value)
                        .collect();
                    eval_op(&self.nodes[i].op, &vals)?
                }
            };
            self.nodes[i].value = value;
        }
        Ok(self.nodes[self.nodes.len() - 1].value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sin_xy_plus_x_sqr() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let xy = g.mul(x, y);
        let x_sqr = g.pow(x, 2.0);
        let sin_xy = g.sin(xy);
        let _f = g.add(sin_xy, x_sqr);

        let inputs = HashMap::from([("x".to_string(), 1.5), ("y".to_string(), 2.0)]);

        // x is ONE node shared by mul and pow
        assert!(g.nodes[xy].inputs.contains(&x));
        assert!(g.nodes[x_sqr].inputs.contains(&x));
        assert_eq!(g.nodes.len(), 6); // 6 nodes total; x isn't copied       

        let result = g.forward(&inputs).expect("forward should succeed");
        let expected = (1.5_f64 * 2.0).sin() + 1.5_f64.powi(2); // sin(3.0) + 2.25

        assert!((result - expected).abs() < 1e-9);
    }

    // A Var whose name is absent from the inputs map must error, not panic.
    #[test]
    fn missing_variable_errors() {
        let mut g = Graph::new();
        let _x = g.var("x".into());

        let inputs = HashMap::new(); // "x" not provided
        let result = g.forward(&inputs);

        assert!(matches!(result, Err(EngineError::UnknownVariable(name)) if name == "x"));
    }

    // div(a, b) with b == 0 must return DivByZero, not silently produce inf.
    #[test]
    fn div_by_zero_errors() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let _q = g.div(x, y);

        let inputs = HashMap::from([("x".to_string(), 1.0), ("y".to_string(), 0.0)]);
        let result = g.forward(&inputs);

        assert!(matches!(result, Err(EngineError::DivByZero)));
    }

    // ln(x) with x <= 0 must return a DomainError, not silently produce NaN.
    #[test]
    fn ln_non_positive_errors() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let _l = g.ln(x);

        let inputs = HashMap::from([("x".to_string(), -1.0)]);
        let result = g.forward(&inputs);

        assert!(matches!(result, Err(EngineError::DomainError(_))));
    }

    // pow(a, k) with a < 0 and non-integer k has no real value → DomainError.
    #[test]
    fn pow_negative_base_fractional_errors() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let _p = g.pow(x, 0.5);

        let inputs = HashMap::from([("x".to_string(), -4.0)]);
        let result = g.forward(&inputs);

        assert!(matches!(result, Err(EngineError::DomainError(_))));
    }
}
