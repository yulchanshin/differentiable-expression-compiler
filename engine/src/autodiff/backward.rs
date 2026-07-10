use crate::graph::arena::Graph;
use crate::graph::node::OpType;
use std::collections::HashMap;

impl Graph {
    pub fn backward(&mut self) -> HashMap<String, f64> {
        for node in &mut self.nodes {
            node.adjoint = 0.0;
        }

        let len: usize = self.nodes.len();
        self.nodes[len - 1].adjoint = 1.0;

        for i in (0..len).rev() {
            let g: f64 = self.nodes[i].adjoint; // ḡ: this node's accumulated adjoint

            match self.nodes[i].op {
                // leaves: no inputs to push to
                OpType::Var(_) | OpType::Const(_) => {}

                OpType::Add => {
                    // add(a, b) = a + b
                    // ∂/∂a = +1  →  ā += ḡ
                    // ∂/∂b = +1  →  b̄ += ḡ
                    let a: usize = self.nodes[i].inputs[0];
                    let b: usize = self.nodes[i].inputs[1];
                    self.nodes[a].adjoint += g;
                    self.nodes[b].adjoint += g;
                }
                OpType::Sub => {
                    // sub(a, b) = a − b
                    // ∂/∂a = +1  →  ā += ḡ
                    // ∂/∂b = −1  →  b̄ += −ḡ  (i.e. b̄ -= ḡ)
                    let a: usize = self.nodes[i].inputs[0];
                    let b: usize = self.nodes[i].inputs[1];
                    self.nodes[a].adjoint += g;
                    self.nodes[b].adjoint -= g;
                }
                OpType::Mul => {
                    // mul(a, b) = a * b
                    // ∂/∂a = b  →  ā += ḡ·b
                    // ∂/∂b = a  →  b̄ += ḡ·a
                    let a: usize = self.nodes[i].inputs[0];
                    let b: usize = self.nodes[i].inputs[1];
                    let av: f64 = self.nodes[a].value;
                    let bv: f64 = self.nodes[b].value;
                    self.nodes[a].adjoint += g * bv;
                    self.nodes[b].adjoint += g * av;
                }
                OpType::Div => {
                    // div(a, b) = a / b
                    // ∂/∂a =  1/b  →  ā += ḡ/b
                    // ∂/∂b = −a/b² →  b̄ += −ḡ·a/(b·b)
                    let a: usize = self.nodes[i].inputs[0];
                    let b: usize = self.nodes[i].inputs[1];
                    let av: f64 = self.nodes[a].value;
                    let bv: f64 = self.nodes[b].value;
                    self.nodes[a].adjoint += g / bv;
                    self.nodes[b].adjoint += -(g * av) / (bv * bv);
                }
                OpType::Neg => {
                    // neg(a) = −a
                    // ∂/∂a = −1  →  ā += −ḡ  (i.e. ā -= ḡ)
                    let a: usize = self.nodes[i].inputs[0];
                    self.nodes[a].adjoint -= g;
                }
                OpType::Pow(n) => {
                    // pow(a, k) = a^k
                    // ∂/∂a = k·a^(k−1)  →  ā += ḡ·k·a^(k−1)
                    let a: usize = self.nodes[i].inputs[0];
                    let av: f64 = self.nodes[a].value;
                    self.nodes[a].adjoint += g * n * av.powf(n - 1.0);
                }
                OpType::Sin => {
                    // sin(a)
                    // ∂/∂a = cos a  →  ā += ḡ·cos a
                    let a: usize = self.nodes[i].inputs[0];
                    let av: f64 = self.nodes[a].value;
                    self.nodes[a].adjoint += g * av.cos()
                }
                OpType::Cos => {
                    // cos(a)
                    // ∂/∂a = −sin a  →  ā += −ḡ·sin a
                    let a: usize = self.nodes[i].inputs[0];
                    let av: f64 = self.nodes[a].value;
                    self.nodes[a].adjoint += (-g) * av.sin();
                }
                OpType::Exp => {
                    // exp(a)
                    // ∂/∂a = exp a  →  ā += ḡ·exp a   (= ḡ·v, this node's own value)
                    let a: usize = self.nodes[i].inputs[0];
                    let av: f64 = self.nodes[a].value;
                    self.nodes[a].adjoint += g * av.exp();
                }
                OpType::Ln => {
                    // ln(a)
                    // ∂/∂a = 1/a  →  ā += ḡ/a
                    let a: usize = self.nodes[i].inputs[0];
                    let av: f64 = self.nodes[a].value;
                    self.nodes[a].adjoint += g / av;
                }
            }
        }

        // extract the gradient from the Var nodes
        let mut grad = HashMap::new();
        for node in &self.nodes {
            if let OpType::Var(name) = &node.op {
                grad.insert(name.clone(), node.adjoint);
            }
        }
        grad
    }
}
