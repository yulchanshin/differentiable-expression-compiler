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
