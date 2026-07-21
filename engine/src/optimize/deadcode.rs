//! Dead-node elimination (DCE): drop nodes unreachable from the output.
//!
//! Constant folding and CSE mutate the arena in place and never delete: a
//! folded input or a redirected duplicate stays resident as an *orphan*,
//! unreachable from the output but still occupying an index (and still
//! needlessly evaluated by the forward pass). DCE is the pass that reclaims
//! them. It is the *one* pass that renumbers the arena, so it runs last in the
//! pipeline (`const_fold -> cse -> dce`) once the earlier passes have stranded
//! everything they are going to strand.
//!
//! ## The algorithm
//! 1. **Mark.** Reverse-reachability from the output. The output is the last
//!    node (`len - 1`), matching the convention the forward pass relies on
//!    (`autodiff::forward` returns `nodes[len - 1]`). Walk input edges backward,
//!    marking every node the output actually depends on.
//! 2. **Renumber.** Assign new indices to the marked nodes in ascending old
//!    order, recorded in a `remap` table.
//! 3. **Compact.** Move the marked nodes into a fresh `Vec`, rewriting each
//!    node's inputs through `remap`.
//!
//! ## Why ascending order is load-bearing
//! The arena's core invariant is "index order is a valid topological order": a
//! node's inputs always sit at lower indices (builder helpers push inputs
//! first), so the forward pass can evaluate in plain index order. Compacting in
//! ascending old-index order preserves that invariant: survivors keep their
//! relative order, so inputs still precede their consumers. It also keeps the
//! output last (it had the highest index and is reachable from itself), so the
//! forward pass keeps returning the right node with no changes.
//!
//! Because DCE renumbers, any index cached before it runs is invalid afterward.
//! That is the price of compaction, and the reason no other pass renumbers.

use crate::graph::arena::Graph;

impl Graph {
    /// Remove every node unreachable from the output (the last node),
    /// renumbering the survivors in ascending order.
    ///
    /// Returns the number of nodes removed, which is `0` when every node is
    /// live. The output value is unchanged: only nodes the output does not
    /// depend on are dropped, and the reachable subgraph is rewired to its new
    /// indices. A redirected CSE duplicate or a folded-away constant leaf is
    /// exactly such an unreachable node, so `const_fold`/`cse` followed by `dce`
    /// actually shrinks the arena.
    pub fn dce(&mut self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        // 1. Mark everything reachable from the output via a backward walk over
        //    input edges. The output is the last node, per the arena convention.
        let root = self.nodes.len() - 1;
        let mut keep = vec![false; self.nodes.len()];
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if keep[i] {
                continue;
            }
            keep[i] = true;
            for &j in &self.nodes[i].inputs {
                stack.push(j);
            }
        }

        // 2. New index for each survivor, assigned in ascending old order so the
        //    topological invariant (inputs before consumers) survives. Dead
        //    slots get a sentinel: no survivor references them, so it never reads.
        let old_len = self.nodes.len();
        let mut remap = vec![usize::MAX; old_len];
        let mut next = 0;
        for (i, &alive) in keep.iter().enumerate() {
            if alive {
                remap[i] = next;
                next += 1;
            }
        }

        // 3. Move survivors into a fresh arena, rewiring their inputs. `mem::take`
        //    lets us consume the old nodes by value (Node isn't Clone) without
        //    holding a borrow of `self`.
        let old = std::mem::take(&mut self.nodes);
        self.nodes.reserve(next);
        for (i, mut node) in old.into_iter().enumerate() {
            if !keep[i] {
                continue;
            }
            for input in node.inputs.iter_mut() {
                *input = remap[*input];
            }
            self.nodes.push(node);
        }

        old_len - self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Every node's inputs must reference strictly lower indices: the topological
    // invariant the forward pass depends on. DCE must preserve it.
    fn assert_topo_invariant(g: &Graph) {
        for (i, node) in g.nodes.iter().enumerate() {
            for &input in &node.inputs {
                assert!(input < i, "node {i} references non-earlier input {input}");
            }
        }
    }

    #[test]
    fn removes_cse_orphan() {
        // x*y + x*y built without hash-consing: CSE redirects one mul to an
        // orphan, and DCE must reclaim exactly that orphan.
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let m1 = g.mul(x, y);
        let m2 = g.mul(x, y);
        let _root = g.add(m1, m2);
        assert_eq!(g.nodes.len(), 5);

        let point = HashMap::from([("x".to_string(), 2.0), ("y".to_string(), 3.0)]);
        let before = g.forward(&point).expect("forward before");

        assert_eq!(g.cse(), 1); // one mul redirected to an orphan
        assert_eq!(g.dce(), 1); // that orphan removed
        assert_eq!(g.nodes.len(), 4); // x, y, mul, add

        let after = g.forward(&point).expect("forward after");
        assert!((before - after).abs() < 1e-9);
        assert_topo_invariant(&g);
    }

    #[test]
    fn removes_constfold_leftovers() {
        // 2*3 + x: const_fold collapses the mul to a Const and orphans the 2
        // and 3 leaves; DCE reclaims both.
        let mut g = Graph::new();
        let two = g.constant(2.0);
        let three = g.constant(3.0);
        let x = g.var("x".into());
        let prod = g.mul(two, three);
        let _root = g.add(prod, x);

        let point = HashMap::from([("x".to_string(), 4.0)]);
        let before = g.forward(&point).expect("forward before");

        g.const_fold();
        let removed = g.dce();
        assert_eq!(removed, 2); // the 2 and the 3

        let after = g.forward(&point).expect("forward after");
        assert!((before - after).abs() < 1e-9);
        assert_topo_invariant(&g);
    }

    #[test]
    fn live_graph_unchanged() {
        // No dead nodes: DCE is a no-op that removes nothing.
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let _root = g.mul(x, y);
        let len_before = g.nodes.len();

        assert_eq!(g.dce(), 0);
        assert_eq!(g.nodes.len(), len_before);
        assert_topo_invariant(&g);
    }

    #[test]
    fn output_stays_last() {
        // After DCE the output must still be the final node so `forward` returns
        // it. Build a graph with a dead branch hanging off nothing reachable.
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let _dead = g.mul(x, y); // never used by the root
        let root = g.add(x, y);
        assert_eq!(root, g.nodes.len() - 1);

        assert_eq!(g.dce(), 1); // the unused mul
        // Root is now the last node and structurally an Add of the two vars.
        let last = g.nodes.len() - 1;
        assert!(matches!(g.nodes[last].op, crate::graph::node::OpType::Add));
        assert_topo_invariant(&g);
    }

    #[test]
    fn empty_graph_is_a_noop() {
        let mut g = Graph::new();
        assert_eq!(g.dce(), 0);
        assert!(g.nodes.is_empty());
    }

    // A multivariable graph with duplicated and commuted subexpressions, built by
    // hand so the duplicates actually exist (the parser would hash-cons them):
    //   (x*y) + (y*x) + sin(x*y) + (x+z) + (z+x)
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
        let _root = g.add(t3, d);
        g
    }

    #[test]
    fn full_pipeline_preserves_value_multivar() {
        use rand::RngExt;

        let mut rng = rand::rng();
        for _ in 0..20 {
            let point: HashMap<String, f64> = ["x", "y", "z"]
                .iter()
                .map(|v| (v.to_string(), rng.random_range(-5.0..5.0)))
                .collect();

            let mut g = build_dupe_graph();
            let before = g.forward(&point).expect("forward before pipeline");

            // The canonical pipeline: fold, share, then reclaim.
            g.const_fold();
            let merged = g.cse();
            let removed = g.dce();
            assert!(merged > 0, "graph should contain duplicates to merge");
            assert!(removed > 0, "CSE should leave orphans for DCE to remove");

            let after = g.forward(&point).expect("forward after pipeline");
            assert!(
                (before - after).abs() < 1e-9,
                "pipeline changed the result: {before} vs {after}"
            );
            assert_topo_invariant(&g);
        }
    }
}
