//arena.rs
//!description will be written when i write this file

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
