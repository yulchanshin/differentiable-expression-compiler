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
    /// variable node holding `∂fᵢ/∂xⱼ` — that is row `i`. So we run one forward
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
