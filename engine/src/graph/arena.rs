//! Arena representation of the computation graph.
//!
//! All nodes live in one `Vec<Node>` that owns them; nodes reference each other
//! by `usize` index, not `Rc`. Indices keep single ownership and avoid the
//! runtime borrow panics an `Rc<RefCell<Node>>` graph would risk.

use crate::graph::node::{Node, OpType};

pub struct Graph {
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

    pub fn push(&mut self, node: Node) -> usize {
        let len: usize = self.nodes.len();
        self.nodes.push(node);
        len
    }

    // All builder helpers funnel through these two: a node starts with zeroed
    // `value`/`adjoint` (filled by the forward/backward passes) and differs only
    // in its op and input list.
    fn push_op(&mut self, op: OpType, inputs: Vec<usize>) -> usize {
        self.push(Node {
            op,
            inputs,
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn var(&mut self, name: String) -> usize {
        self.push_op(OpType::Var(name), vec![])
    }

    pub fn constant(&mut self, val: f64) -> usize {
        self.push_op(OpType::Const(val), vec![])
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        self.push_op(OpType::Add, vec![a, b])
    }

    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        self.push_op(OpType::Sub, vec![a, b])
    }

    pub fn div(&mut self, a: usize, b: usize) -> usize {
        self.push_op(OpType::Div, vec![a, b])
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        self.push_op(OpType::Mul, vec![a, b])
    }

    pub fn neg(&mut self, a: usize) -> usize {
        self.push_op(OpType::Neg, vec![a])
    }

    pub fn pow(&mut self, a: usize, exp: f64) -> usize {
        self.push_op(OpType::Pow(exp), vec![a])
    }

    pub fn sin(&mut self, a: usize) -> usize {
        self.push_op(OpType::Sin, vec![a])
    }

    pub fn cos(&mut self, a: usize) -> usize {
        self.push_op(OpType::Cos, vec![a])
    }

    pub fn exp(&mut self, a: usize) -> usize {
        self.push_op(OpType::Exp, vec![a])
    }

    pub fn ln(&mut self, a: usize) -> usize {
        self.push_op(OpType::Ln, vec![a])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_x_times_y() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let z = g.mul(x, y);

        assert_eq!(x, 0);
        assert_eq!(y, 1);
        assert_eq!(z, 2);
        assert_eq!(g.nodes[z].inputs, vec![x, y]);
        assert_eq!(g.nodes.len(), 3);
    }

    #[test]
    fn shared_node_appears_twice() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let z = g.var("z".into());

        let m1 = g.mul(x, y); // x * y
        let m2 = g.mul(x, z); // x * z

        assert!(g.nodes[m1].inputs.contains(&x)); // x is an input of m1
        assert!(g.nodes[m2].inputs.contains(&x)); // x is ALSO an input of m2
    }

    #[test]
    fn sin_multivar() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let _xy = g.mul(x, y);
        let _x_sqr = g.pow(x, 2.0);
    }
}
