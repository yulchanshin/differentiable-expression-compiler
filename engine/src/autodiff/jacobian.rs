//! Jacobian assembly for vector functions `f: ℝⁿ → ℝᵐ`.
//!
//! Runs one backward pass per output (seeding that output's adjoint to 1) to
//! collect each row `∂fᵢ/∂xⱼ`, reusing a single forward pass across all rows.
//! See [`Graph::jacobian`] for the full derivation.

use crate::error::EngineError;
use crate::graph::arena::Graph;
use std::collections::HashMap;

impl Graph {
    /// Assemble the Jacobian of a vector function `f: ℝⁿ → ℝᵐ`.
    ///
    /// For `f(x) = (f₁(x), …, f_m(x))` over variables `x = (x₁, …, xₙ)`, the
    /// Jacobian is the m×n matrix of first partials
    ///
    /// ```text
    ///        ⎡ ∂f₁/∂x₁  ∂f₁/∂x₂  …  ∂f₁/∂xₙ ⎤
    ///   J =  ⎢ ∂f₂/∂x₁  ∂f₂/∂x₂  …  ∂f₂/∂xₙ ⎥ ,   J[i][j] = ∂fᵢ/∂xⱼ .
    ///        ⎣   ⋮         ⋮      ⋱     ⋮   ⎦
    /// ```
    ///
    /// Reverse-mode AD computes **one row per backward pass**: seeding output
    /// `i`'s adjoint to 1 (all others 0) and propagating backward leaves each
    /// variable node holding `∂fᵢ/∂xⱼ`, i.e. row `i`. So we run one forward
    /// pass (done by the caller, shared across all rows) and `m = outputs.len()`
    /// backward passes, one per output.
    ///
    /// `outputs[i]` is the node index of `fᵢ` (fixing row order); `vars[j]` is
    /// the name of `xⱼ` (fixing column order, since the gradient map is
    /// unordered). Assumes [`forward`](Self::forward) has already populated node
    /// values. Returns the dense m×n matrix, or the first `EngineError` a
    /// backward pass raises.
    pub fn jacobian(
        &mut self,
        outputs: &[usize],
        vars: &[String],
    ) -> Result<Vec<Vec<f64>>, EngineError> {
        let mut jac: Vec<Vec<f64>> = Vec::new();
        for &output in outputs {
            let grad: HashMap<String, f64> = self.backward(output)?;
            let row: Vec<f64> = vars.iter().map(|v| grad[v]).collect();
            jac.push(row);
        }
        Ok(jac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    // polar → cartesian:  x = r·cosθ,  y = r·sinθ
    // Analytic Jacobian:
    //   ∂x/∂r = cosθ     ∂x/∂θ = −r·sinθ
    //   ∂y/∂r = sinθ     ∂y/∂θ =  r·cosθ
    #[test]
    fn polar_to_cartesian() {
        let mut g = Graph::new();

        let r: usize = g.var("r".into());
        let theta: usize = g.var("theta".into());

        let cos_theta: usize = g.cos(theta);
        let sin_theta: usize = g.sin(theta);

        let r_cos_theta: usize = g.mul(r, cos_theta);
        let r_sin_theta: usize = g.mul(r, sin_theta);

        let r_value: f64 = 2.0;
        let theta_value: f64 = 0.5;

        let inputs: HashMap<String, f64> =
            HashMap::from([("r".to_string(), r_value), ("theta".to_string(), theta_value)]);
        g.forward(&inputs).expect("forward should succeed");

        let outputs: [usize; 2] = [r_cos_theta, r_sin_theta];
        let variables: [String; 2] = ["r".to_string(), "theta".to_string()];
        let jac: Vec<Vec<f64>> = g
            .jacobian(&outputs, &variables)
            .expect("jacobian should succeed");

        let dxdr: f64 = theta_value.cos();
        let dydr: f64 = theta_value.sin();
        let dxdt: f64 = -r_value * theta_value.sin();
        let dydt: f64 = r_value * theta_value.cos();

        assert!((dxdr - jac[0][0]).abs() < TOLERANCE);
        assert!((dydr - jac[1][0]).abs() < TOLERANCE);
        assert!((dxdt - jac[0][1]).abs() < TOLERANCE);
        assert!((dydt - jac[1][1]).abs() < TOLERANCE);
    }
}
