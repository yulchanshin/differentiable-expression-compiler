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

struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

    fn push(&mut self, node: Node) -> usize {
        let len: usize = self.nodes.len();
        self.nodes.push(node);
        len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_x_times_y() {
        let mut g = Graph::new();
        let x = g.push(Node {
            op: OpType::Var,
            inputs: vec![],
            value: 0.0,
            adjoint: 0.0,
        });

        let y = g.push(Node {
            op: OpType::Var,
            inputs: vec![],
            value: 0.0,
            adjoint: 0.0,
        });

        let z = g.push(Node {
            op: OpType::Mul,
            inputs: vec![x, y],
            value: 0.0,
            adjoint: 0.0,
        });

        assert_eq!(0, x);
        assert_eq!(1, y);
        assert_eq!(2, z);
        assert_eq!(vec![x, y], g.nodes[2].inputs);
        assert_eq!(3, g.nodes.len());
    }

    #[test]
    fn shared_node_appears_twice() {
        let mut g = Graph::new();
        let x = g.push(Node {
            op: OpType::Var,
            inputs: vec![],
            value: 0.0,
            adjoint: 0.0,
        }); // index 0
        let y = g.push(Node {
            op: OpType::Var,
            inputs: vec![],
            value: 0.0,
            adjoint: 0.0,
        }); // index 1
        let z = g.push(Node {
            op: OpType::Var,
            inputs: vec![],
            value: 0.0,
            adjoint: 0.0,
        }); // index 2

        let m1 = g.push(Node {
            op: OpType::Mul,
            inputs: vec![x, y],
            value: 0.0,
            adjoint: 0.0,
        }); // x * y
        let m2 = g.push(Node {
            op: OpType::Mul,
            inputs: vec![x, z],
            value: 0.0,
            adjoint: 0.0,
        });
        assert!(g.nodes[m1].inputs.contains(&x)); // x is an input of m1
        assert!(g.nodes[m2].inputs.contains(&x)); // x is ALSO an input of m2       }); // x * z
    }
}
