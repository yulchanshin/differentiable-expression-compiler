//! Common-subexpression elimination (CSE): merge structurally identical nodes.
//!
//! Lowering already hash-conses a single parse; CSE applies the same idea to an
//! arbitrary graph as a visible pass, so it also catches duplicates built by
//! hand or produced by other passes. One forward sweep (index order is
//! topological): rewrite each node's inputs to their canonical representative,
//! key it via [`NodeKey::canonical`] (which sorts commutative operands so `a*b`
//! and `b*a` collapse), then reuse an existing representative or become one.
//!
//! Like constant folding it rewrites in place and never removes nodes; redirected
//! duplicates become unreachable orphans that dead-node elimination reclaims.

use std::collections::HashMap;

use crate::graph::arena::Graph;
use crate::graph::key::NodeKey;

impl Graph {
    /// Merge structurally identical nodes into one canonical node, in place.
    ///
    /// Returns the number of nodes eliminated (redirected to an earlier
    /// representative), which is `0` when the graph has no duplicates. Commutative
    /// operands are canonicalized, so `a*b` and `b*a` count as identical.
    pub fn cse(&mut self) -> usize {
        // memo: structural key -> index of the node that realizes it.
        let mut memo: HashMap<NodeKey, usize> = HashMap::new();
        // remap[i] = canonical representative of node i. Starts as identity and
        // is filled in as the sweep decides each node's fate.
        let mut remap: Vec<usize> = (0..self.nodes.len()).collect();
        let mut eliminated: usize = 0;

        for i in 0..self.nodes.len() {
            // 1. Point this node's inputs at their canonical reps. Inputs sit at
            //    lower indices, so `remap` already holds their final answer.
            let inputs: Vec<usize> = self.nodes[i]
                .inputs
                .iter()
                .map(|&j| remap[j])
                .collect();
            self.nodes[i].inputs = inputs;

            // 2. Key on (op, canonicalized inputs).
            let key = NodeKey::canonical(&self.nodes[i].op, &self.nodes[i].inputs);

            // 3. Dedup: reuse the existing representative, or become one.
            match memo.get(&key) {
                Some(&canon) => {
                    remap[i] = canon;
                    eliminated += 1;
                }
                None => {
                    memo.insert(key, i);
                }
            }
        }

        eliminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Count nodes reachable from `root`. After CSE the arena still holds the
    // redirected duplicates as orphans, so `nodes.len()` does not shrink until
    // dead-node elimination runs; the reachable set is the "minimal shared form"
    // the acceptance criterion cares about.
    fn reachable_count(g: &Graph, root: usize) -> usize {
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

    #[test]
    fn duplicate_subexpr_collapses() {
        // x*y + x*y, built without hash-consing: two separate mul nodes.
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let m1 = g.mul(x, y);
        let m2 = g.mul(x, y);
        let root = g.add(m1, m2);
        assert_eq!(g.nodes.len(), 5); // x, y, m1, m2, add

        let eliminated = g.cse();
        assert_eq!(eliminated, 1); // one duplicate mul redirected

        // The add's two inputs are now the same canonical mul.
        assert_eq!(g.nodes[root].inputs[0], g.nodes[root].inputs[1]);
        // Minimal shared form: x, y, mul, add.
        assert_eq!(reachable_count(&g, root), 4);
    }

    #[test]
    fn commutative_operands_canonicalized() {
        // x*y and y*x differ only in operand order, so CSE must merge them.
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let m1 = g.mul(x, y);
        let m2 = g.mul(y, x);
        let root = g.add(m1, m2);

        let eliminated = g.cse();
        assert_eq!(eliminated, 1);
        assert_eq!(g.nodes[root].inputs[0], g.nodes[root].inputs[1]);
        assert_eq!(reachable_count(&g, root), 4);
    }

    #[test]
    fn non_commutative_operands_not_merged() {
        // x-y and y-x are different values; subtraction does not commute.
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let s1 = g.sub(x, y);
        let s2 = g.sub(y, x);
        let root = g.add(s1, s2);

        let eliminated = g.cse();
        assert_eq!(eliminated, 0);
        assert_ne!(g.nodes[root].inputs[0], g.nodes[root].inputs[1]);
        assert_eq!(reachable_count(&g, root), 5); // x, y, s1, s2, add
    }

    #[test]
    fn identical_constants_dedup() {
        // Two separately-built Const(2.0) nodes collapse to one (bit-exact key).
        let mut g = Graph::new();
        let c1 = g.constant(2.0);
        let c2 = g.constant(2.0);
        let root = g.add(c1, c2);

        let eliminated = g.cse();
        assert_eq!(eliminated, 1);
        assert_eq!(g.nodes[root].inputs[0], g.nodes[root].inputs[1]);
        assert_eq!(reachable_count(&g, root), 2); // one const, one add
    }

    #[test]
    fn empty_graph_is_a_noop() {
        let mut g = Graph::new();
        assert_eq!(g.cse(), 0);
    }

    // A graph with duplicated and commuted subexpressions, built by hand so the
    // duplicates actually exist (the parser would hash-cons them away):
    //   (x*y) + (y*x) + sin(x*y) + (x+z) + (z+x)
    fn build_dupe_graph() -> (Graph, usize) {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let z = g.var("z".into());

        let a = g.mul(x, y); // x*y
        let b = g.mul(y, x); // y*x  -- commuted duplicate of a
        let s = g.sin(a); // sin(x*y)
        let c = g.add(x, z); // x+z
        let d = g.add(z, x); // z+x  -- commuted duplicate of c

        let t1 = g.add(a, b);
        let t2 = g.add(t1, s);
        let t3 = g.add(t2, c);
        let root = g.add(t3, d);
        (g, root)
    }

    #[test]
    fn cse_preserves_forward_value() {
        use rand::RngExt;
        use std::collections::HashMap;

        let mut rng = rand::rng();
        for _ in 0..20 {
            let point: HashMap<String, f64> = ["x", "y", "z"]
                .iter()
                .map(|v| (v.to_string(), rng.random_range(-5.0..5.0)))
                .collect();

            // Evaluate before and after CSE on the same graph; commuting and
            // sharing operands must not change the computed output.
            let (mut g, _root) = build_dupe_graph();
            let before = g.forward(&point).expect("forward before CSE");
            let eliminated = g.cse();
            assert!(eliminated > 0, "graph should contain duplicates to remove");
            let after = g.forward(&point).expect("forward after CSE");

            assert!(
                (before - after).abs() < 1e-9,
                "CSE changed the result: {before} vs {after}"
            );
        }
    }
}
