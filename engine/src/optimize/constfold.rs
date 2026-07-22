//! Constant folding: evaluate all-constant subgraphs at compile time.
//!
//! When every input of a node is a `Const`, its value is fixed, so we replace
//! the node with a single `Const`. One forward sweep suffices (index order is
//! topological), so a folded constant is visible to its consumers in the same
//! pass. Folding rewrites in place and does not compact; the former inputs
//! become unreachable and are reclaimed by dead-node elimination. A node whose
//! evaluation would fail a domain check (div by zero, `ln` of a non-positive
//! value, `pow` of a negative base to a fractional power) is left untouched so
//! the error still surfaces at real evaluation time.

use crate::graph::arena::Graph;
use crate::graph::node::OpType;
use crate::ops::eval::eval_op;

impl Graph {
    /// Fold every all-constant node into a single `Const` in one topological
    /// sweep, mutating the arena in place.
    ///
    /// Returns the number of nodes folded, which is `0` when the graph contains
    /// nothing foldable. A node is folded only when all of its inputs are
    /// already `Const` and its evaluation stays within domain; otherwise it is
    /// left as is.
    pub fn const_fold(&mut self) -> usize {
        let mut count: usize = 0;
        for i in 0..self.nodes.len() {
            if self.nodes[i].inputs.is_empty() {
                continue;
            }

            let mut vals: Vec<f64> = Vec::with_capacity(self.nodes[i].inputs.len());
            for k in 0..self.nodes[i].inputs.len() {
                //checking if input node is a constant
                match &self.nodes[self.nodes[i].inputs[k]].op {
                    OpType::Const(c) => vals.push(*c),
                    _ => break,
                }
            }

            //if not all inputs were const, this evaluates to true
            if vals.len() != self.nodes[i].inputs.len() {
                continue;
            }

            // Evaluate with the shared op semantics; a domain failure means we
            // leave the node unfolded so the error still surfaces at real
            // evaluation time rather than baking an inf/NaN constant.
            let result = match eval_op(&self.nodes[i].op, &vals) {
                Ok(v) => v,
                Err(_) => continue,
            };

            self.nodes[i].op = OpType::Const(result);
            self.nodes[i].inputs.clear();
            count += 1;
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Count nodes reachable from the output (the last node), walking inputs.
    // const_fold rewrites in place and leaves orphaned inputs in the arena, so
    // the *reachable* count is what shrinks, not `nodes.len()` (DCE compacts
    // the arena later, in TICKET-402).
    fn reachable(g: &Graph) -> usize {
        let mut seen = vec![false; g.nodes.len()];
        let mut stack = vec![g.nodes.len() - 1];
        while let Some(i) = stack.pop() {
            if seen[i] {
                continue;
            }
            seen[i] = true;
            for &k in &g.nodes[i].inputs {
                stack.push(k);
            }
        }
        seen.iter().filter(|&&b| b).count()
    }

    // A tiny deterministic LCG so the "random" property-test points are
    // reproducible without pulling in the `rand` crate.
    fn sample_points(n: usize) -> Vec<f64> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // top 53 bits -> [0, 1), remapped to [-10, 10)
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 20.0 - 10.0
            })
            .collect()
    }

    // AC: `x + 2*3` folds `2*3 -> 6`, leaving `x + 6`. One node folds and the
    // reachable count drops from 5 (x, 2, 3, mul, add) to 3 (x, 6, add).
    #[test]
    fn folds_constant_subexpression() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let two = g.constant(2.0);
        let three = g.constant(3.0);
        let prod = g.mul(two, three);
        let root = g.add(x, prod);

        assert_eq!(reachable(&g), 5);

        let folded = g.const_fold();

        assert_eq!(folded, 1);
        assert!(matches!(g.nodes[prod].op, OpType::Const(c) if c == 6.0));
        assert!(g.nodes[prod].inputs.is_empty());
        // The add survives: one input (x) is a Var, so it isn't all-constant.
        assert!(matches!(g.nodes[root].op, OpType::Add));
        assert_eq!(reachable(&g), 3);
    }

    // AC property test: for f(x) = x*(2*3) + (4-1), the forward value at many
    // points is identical before and after folding.
    #[test]
    fn fold_preserves_values() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let two = g.constant(2.0);
        let three = g.constant(3.0);
        let c1 = g.mul(two, three); // 6
        let four = g.constant(4.0);
        let one = g.constant(1.0);
        let c2 = g.sub(four, one); // 3
        let term = g.mul(x, c1);
        let _root = g.add(term, c2); // x*6 + 3

        let pts = sample_points(1000);
        let before: Vec<f64> = pts
            .iter()
            .map(|&v| {
                g.forward(&HashMap::from([("x".to_string(), v)]))
                    .expect("forward should succeed")
            })
            .collect();

        let folded = g.const_fold();
        assert!(folded >= 2); // c1 and c2 both fold

        for (&v, &want) in pts.iter().zip(&before) {
            let got = g
                .forward(&HashMap::from([("x".to_string(), v)]))
                .expect("forward should succeed");
            assert!(
                (got - want).abs() < 1e-12,
                "fold changed value at x={v}: {got} vs {want}"
            );
        }
    }

    // A node whose evaluation would fail a domain check (1/0) is left unfolded
    // so the error still surfaces at real evaluation time.
    #[test]
    fn domain_failure_left_unfolded() {
        let mut g = Graph::new();
        let a = g.constant(1.0);
        let b = g.constant(0.0);
        let q = g.div(a, b);

        let folded = g.const_fold();

        assert_eq!(folded, 0);
        assert!(matches!(g.nodes[q].op, OpType::Div));
        assert!(!g.nodes[q].inputs.is_empty());
    }
}
