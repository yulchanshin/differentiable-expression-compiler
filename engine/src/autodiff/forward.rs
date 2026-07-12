use crate::error::EngineError;
use crate::graph::arena::Graph;
use crate::graph::node::OpType;
use std::collections::HashMap;

// Index order is a valid topological order: builder helpers always push a
// node's inputs before the node itself, so every input sits at a lower index.
impl Graph {
    pub fn forward(&mut self, inputs: &HashMap<String, f64>) -> Result<f64, EngineError> {
        for i in 0..self.nodes.len() {
            let value = match &self.nodes[i].op {
                OpType::Const(c) => *c,
                OpType::Var(name) => *inputs
                    .get(name)
                    .ok_or_else(|| EngineError::UnknownVariable(name.clone()))?,
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
                    if b == 0.0 {
                        return Err(EngineError::DivByZero);
                    }
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
                    if a < 0.0 && n.fract() != 0.0 {
                        return Err(EngineError::DomainError(format!(
                            "pow with negative base {a} and non-integer exponent {n}"
                        )));
                    }
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
                    if a <= 0.0 {
                        return Err(EngineError::DomainError(format!(
                            "ln requires x > 0, but got {a}"
                        )));
                    }
                    a.ln()
                }
            };
            self.nodes[i].value = value;
        }
        Ok(self.nodes[self.nodes.len() - 1].value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sin_xy_plus_x_sqr() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let xy = g.mul(x, y);
        let x_sqr = g.pow(x, 2.0);
        let sin_xy = g.sin(xy);
        let _f = g.add(sin_xy, x_sqr);

        let inputs = HashMap::from([("x".to_string(), 1.5), ("y".to_string(), 2.0)]);

        // x is ONE node shared by mul and pow
        assert!(g.nodes[xy].inputs.contains(&x));
        assert!(g.nodes[x_sqr].inputs.contains(&x));
        assert_eq!(g.nodes.len(), 6); // 6 nodes total; x isn't copied       

        let result = g.forward(&inputs).expect("forward should succeed");
        let expected = (1.5_f64 * 2.0).sin() + 1.5_f64.powi(2); // sin(3.0) + 2.25

        assert!((result - expected).abs() < 1e-9);
    }
}
