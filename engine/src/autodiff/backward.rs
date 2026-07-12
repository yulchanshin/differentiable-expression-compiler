use crate::error::EngineError;
use crate::graph::arena::Graph;
use crate::graph::node::OpType;
use std::collections::HashMap;

// Reverse-mode autodiff: after a forward pass fills each node's value, walk
// nodes in reverse index order (a reverse topological order) and push each
// op's local derivative into its inputs' adjoints, accumulating with += so a
// shared node sums every path. Each var node then holds its partial ∂f/∂var.
impl Graph {
    pub fn backward(&mut self) -> Result<HashMap<String, f64>, EngineError> {
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
                    // a^(k−1) blows up at a = 0 when k < 1 (e.g. d/dx √x at 0),
                    // even though the forward value a^k was finite.
                    if av == 0.0 && n < 1.0 {
                        return Err(EngineError::DomainError(format!(
                            "pow derivative undefined at base 0 with exponent {n} (< 1)"
                        )));
                    }
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
        Ok(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gradient_f() {
        //sin(xy) + x^2
        let mut g = Graph::new();
        let v1: usize = g.var("x".into());
        let v2: usize = g.var("y".into());
        let v1v2: usize = g.mul(v1, v2);
        let sin_v1v2: usize = g.sin(v1v2);
        let v1_sqr: usize = g.pow(v1, 2.0);
        let f: usize = g.add(sin_v1v2, v1_sqr);

        let inputs: HashMap<String, f64> =
            HashMap::from([("x".to_string(), 1.5), ("y".to_string(), 2.0)]);
        g.forward(&inputs).expect("forward should succeed");
        let grad: HashMap<String, f64> = g.backward().expect("backward should succeed");

        // ∂f/∂x = y·cos(xy) + 2x   ∂f/∂y = x·cos(xy)
        // x is a SHARED node (feeds x*y and x^2) so ∂f/∂x sums both paths.
        let xy: f64 = 1.5 * 2.0;
        let expected_dx: f64 = 2.0 * xy.cos() + 2.0 * 1.5;
        let expected_dy: f64 = 1.5 * xy.cos();

        assert!((grad["x"] - expected_dx).abs() < 1e-6);
        assert!((grad["y"] - expected_dy).abs() < 1e-6);

        // one backward pass yields the full gradient
        assert_eq!(grad.len(), 2);

        let _ = f; // output node index, unused
    }

    // A variable used in two terms must SUM both contributions.
    // f(x) = x + x  ⇒  ∂f/∂x = 2 (would be 1 if += were an overwriting =).
    #[test]
    fn shared_var_sums_both_paths() {
        let mut g = Graph::new();
        let x: usize = g.var("x".into());
        let _f: usize = g.add(x, x);

        let inputs: HashMap<String, f64> = HashMap::from([("x".to_string(), 3.0)]);
        g.forward(&inputs).expect("forward should succeed");
        let grad: HashMap<String, f64> = g.backward().expect("backward should succeed");

        assert_eq!(grad["x"], 2.0);
    }

    // sqrt(x) = x^0.5: forward at x = 0 is fine (= 0), but the derivative
    // 0.5·x^(-0.5) is infinite there. backward must return a DomainError.
    #[test]
    fn pow_derivative_at_zero_errors() {
        let mut g = Graph::new();
        let x: usize = g.var("x".into());
        let _p: usize = g.pow(x, 0.5);

        let inputs: HashMap<String, f64> = HashMap::from([("x".to_string(), 0.0)]);
        g.forward(&inputs).expect("forward should succeed at x = 0");
        let result = g.backward();

        assert!(matches!(result, Err(EngineError::DomainError(_))));
    }
}
