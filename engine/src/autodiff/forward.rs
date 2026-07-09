use crate::graph::arena::Graph;
use crate::graph::node::OpType;
use std::collections::HashMap;

// Index order is a valid topological order: builder helpers always push a
// node's inputs before the node itself, so every input sits at a lower index.
impl Graph {
    fn forward(&mut self, inputs: &HashMap<String, f64>) -> f64 {
        for i in 0..self.nodes.len() {
            let value = match &self.nodes[i].op {
                OpType::Const(c) => *c,
                OpType::Var(name) => inputs[name],
                OpType::Add => {
                    let a: f64 = self.nodes[self.nodes[i].inputs[0]].value;
                    let b: f64 = self.nodes[self.nodes[i].inputs[1]].value;
                    a + b
                }
                OpType::Sub => {
                    let a: f64 = self.nodes[self.nodes[i].inputs[0]].value;
                    let b: f64 = self.nodes[self.nodes[i].inputs[1]].value;
                    a - b
                }
                OpType::Div => {
                    let a: f64 = self.nodes[self.nodes[i].inputs[0]].value;
                    let b: f64 = self.nodes[self.nodes[i].inputs[1]].value;
                    a / b
                }
                OpType::Mul => {
                    let a: f64 = self.nodes[self.nodes[i].inputs[0]].value;
                    let b: f64 = self.nodes[self.nodes[i].inputs[1]].value;
                    a * b
                }
                OpType::Neg => {
                    let a: f64 = self.nodes[self.nodes[i].inputs[0]].value;
                    -1.0 * a
                }
                OpType::Pow(n) => {
                    let a: f64 = self.nodes[self.nodes[i].inputs[0]].value;
                    a.powf(*n)
                }
                OpType::Sin => {
                    let a: f64 = self.nodes[self.nodes[i].inputs[0]].value;
                    a.sin()
                }
                OpType::Cos => {
                    let a: f64 = self.nodes[self.nodes[i].inputs[0]].value;
                    a.cos()
                }
                OpType::Exp => {
                    let a: f64 = self.nodes[self.nodes[i].inputs[0]].value;
                    a.exp()
                }
                OpType::Ln => {
                    let a: f64 = self.nodes[self.nodes[i].inputs[0]].value;
                    a.ln()
                }
            };
            self.nodes[i].value = value;
        }
        self.nodes[self.nodes.len() - 1].value
    }
}
