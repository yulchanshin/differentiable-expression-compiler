//! Constant folding: evaluate all-constant subgraphs at compile time.
//!
//! If every input of a node is a `Const`, the node's value is fixed and can be
//! computed now, replacing the node with a single `Const`. This shrinks the
//! graph and lets constants ripple upward so later consumers fold too.
//!
//! A single forward sweep suffices, no fixpoint loop needed. Index order is a
//! valid topological order (builder helpers push a node's inputs before the
//! node itself), so by the time the sweep reaches node `i`, any input that
//! could fold already has. A freshly folded `Const` is therefore visible to
//! every consumer downstream of it in the same pass.
//!
//! Folding rewrites nodes in place; it does not compact the arena. The former
//! input nodes (e.g. the `2` and `3` in `2 * 3`) stay resident but become
//! unreachable, and are reclaimed later by dead-node elimination. Nodes whose
//! evaluation would fail a domain check (division by zero, `ln` of a
//! non-positive value, `pow` of a negative base to a fractional exponent) are
//! left untouched so the error still surfaces at real evaluation time rather
//! than being baked into an `inf`/`NaN` constant.

use crate::graph::arena::Graph;
use crate::graph::node::OpType;

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
                match self.nodes[self.nodes[i].inputs[k]].op {
                    OpType::Const(c) => vals.push(c),
                    _ => break,
                }
            }

            //if not all inputs were const, this evaluates to true
            if (vals.len() != self.nodes[i].inputs.len()) {
                continue;
            }

            let result: f64 = match &self.nodes[i].op {
                OpType::Add => vals[0] + vals[1],
                OpType::Sub => vals[0] - vals[1],
                OpType::Mul => vals[0] * vals[1],
                OpType::Div => vals[0] / vals[1],
                OpType::Neg => -vals[0],
                OpType::Pow(n) => {
                    if vals[0] < 0.0 && n.fract() != 0.0 { 
                        continue;
                    }
                    vals[0].powf(*n) 
                }
                OpType::Sin => vals[0].sin(),
                OpType::Cos => vals[0].cos(),
                OpType::Exp => vals[0].exp(),
                OpType::Ln  => {
                    if vals[0] <= 0.0 { 
                        continue; 
                    } 
                    vals[0].ln() 
                }
                _ => continue,
            }
            self.nodes[i].op = OpType::Const(result);
            self.nodes[i].inputs.clear();
            count += 1;
        }
        count
    }
}
