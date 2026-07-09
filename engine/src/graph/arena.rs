//! Arena Representation of the computation Graph
//!
//! All nodes of a graph live in one `Vec<Node>` that owns them.
//! Nodes refer to each other by `usize` index and not by `Rc`.
//!
//! Why indices instead of `Rc<RefCell<Node>>`:
//!
//! `Rc` allows the nodes to have multiple owners and `RefCell` allows
//! these `Rc` nodes to be mutable. Therefore, there is a chance of a
//! runtime panic happening due to `RefCell`'s runtime borrow checking.
//! With `Vec<Node>` this shouldn't be an issue as we enforce single ownership.

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

    pub fn var(&mut self, name: String) -> usize {
        self.push(Node {
            op: OpType::Var(name),
            inputs: vec![],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn constant(&mut self, val: f64) -> usize {
        self.push(Node {
            op: OpType::Const(val),
            inputs: vec![],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        self.push(Node {
            op: OpType::Add,
            inputs: vec![a, b],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        self.push(Node {
            op: OpType::Sub,
            inputs: vec![a, b],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn div(&mut self, a: usize, b: usize) -> usize {
        self.push(Node {
            op: OpType::Div,
            inputs: vec![a, b],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        self.push(Node {
            op: OpType::Mul,
            inputs: vec![a, b],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn neg(&mut self, a: usize) -> usize {
        self.push(Node {
            op: OpType::Neg,
            inputs: vec![a],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn pow(&mut self, a: usize, exp: f64) -> usize {
        self.push(Node {
            op: OpType::Pow(exp),
            inputs: vec![a],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn sin(&mut self, a: usize) -> usize {
        self.push(Node {
            op: OpType::Sin,
            inputs: vec![a],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn cos(&mut self, a: usize) -> usize {
        self.push(Node {
            op: OpType::Cos,
            inputs: vec![a],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn exp(&mut self, a: usize) -> usize {
        self.push(Node {
            op: OpType::Exp,
            inputs: vec![a],
            value: 0.0,
            adjoint: 0.0,
        })
    }

    pub fn ln(&mut self, a: usize) -> usize {
        self.push(Node {
            op: OpType::Ln,
            inputs: vec![a],
            value: 0.0,
            adjoint: 0.0,
        })
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
