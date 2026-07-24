//! Inverse kinematics for a planar n-link arm via damped least squares.
//!
//! Forward kinematics `p(θ) = (x, y)` is built as an engine expression graph
//! (cumulative joint-angle sums feeding cos/sin), so the engine's own Jacobian
//! `J = ∂p/∂θ` (2×n) drives the solve for free. Each iteration takes the DLS
//! step `Δθ = Jᵀ(JJᵀ + λ²I)⁻¹ e` toward the tip-space error `e = target − p`.
//! The `λ²I` damping keeps the 2×2 system `(JJᵀ + λ²I) y = e` well-conditioned
//! at singular poses (arm fully extended), where the raw pseudoinverse diverges.
//! That small system is the LU solver's one consumer.

use crate::error::EngineError;
use crate::graph::arena::Graph;
use crate::linalg::lu::{lu_decompose, lu_solve};
use std::collections::HashMap;

/// One solver step, recorded for convergence plots / streaming.
pub struct Iteration {
    pub theta: Vec<f64>,
    pub tip: (f64, f64),
    pub error_norm: f64,
}

/// Outcome of [`IkArm::solve`].
pub struct IkResult {
    pub angles: Vec<f64>, // final joint angles (== last history entry's theta)
    pub reached: bool,    // error fell below `tol` before `max_iters` steps
    pub history: Vec<Iteration>,
}

/// A planar n-link arm whose forward kinematics live in an engine graph.
pub struct IkArm {
    graph: Graph,
    theta_names: Vec<String>, // fixes the Jacobian's column order
    x_node: usize,
    y_node: usize,
    lambda: f64, // DLS damping factor λ
}

impl IkArm {
    /// Build an arm from its link lengths (base → tip) and damping factor `λ`.
    pub fn new(link_lengths: &[f64], lambda: f64) -> Self {
        let mut graph = Graph::new();
        let n = link_lengths.len();

        let theta_names: Vec<String> = (0..n).map(|i| format!("theta{i}")).collect();
        let thetas: Vec<usize> = theta_names
            .iter()
            .map(|name| graph.var(name.clone()))
            .collect();

        // x = Σ Lₖ·cos(θ₀+…+θₖ),  y = Σ Lₖ·sin(θ₀+…+θₖ). `angle` accumulates the
        // cumulative joint-angle sum as we walk outward from the base.
        let mut x_node = graph.constant(0.0);
        let mut y_node = graph.constant(0.0);
        let mut angle = thetas[0];
        for k in 0..n {
            if k > 0 {
                angle = graph.add(angle, thetas[k]);
            }
            let len = graph.constant(link_lengths[k]);
            let cos = graph.cos(angle);
            let sin = graph.sin(angle);
            let lx = graph.mul(len, cos);
            let ly = graph.mul(len, sin);
            x_node = graph.add(x_node, lx);
            y_node = graph.add(y_node, ly);
        }

        Self {
            graph,
            theta_names,
            x_node,
            y_node,
            lambda,
        }
    }

    /// Drive the tip toward `target` from initial angles `theta0` using DLS.
    ///
    /// Stops when `‖target − p(θ)‖ < tol` (`reached = true`) or after
    /// `max_iters` update steps. The final `angles` always match the last
    /// history entry, so callers can trust either.
    pub fn solve(
        &mut self,
        target: (f64, f64),
        theta0: &[f64],
        tol: f64,
        max_iters: usize,
    ) -> Result<IkResult, EngineError> {
        let mut theta = theta0.to_vec();
        let mut history = Vec::new();
        let lambda_sq = self.lambda * self.lambda;

        // `0..=max_iters`: one evaluation per step plus a final evaluate-only
        // pass, so the recorded state always matches the returned angles.
        for step in 0..=max_iters {
            // Forward kinematics: populate node values, read the tip off the graph.
            let inputs: HashMap<String, f64> = self
                .theta_names
                .iter()
                .cloned()
                .zip(theta.iter().copied())
                .collect();
            self.graph.forward(&inputs)?;
            let tip = (
                self.graph.nodes[self.x_node].value,
                self.graph.nodes[self.y_node].value,
            );

            let e = [target.0 - tip.0, target.1 - tip.1];
            let error_norm = e[0].hypot(e[1]);
            history.push(Iteration {
                theta: theta.clone(),
                tip,
                error_norm,
            });
            if error_norm < tol {
                return Ok(IkResult {
                    angles: theta,
                    reached: true,
                    history,
                });
            }
            if step == max_iters {
                break; // out of steps; last entry recorded above
            }

            // J is 2×n, so the DLS system (J Jᵀ + λ²I) is always 2×2 for a
            // planar tip, regardless of joint count.
            let j = self
                .graph
                .jacobian(&[self.x_node, self.y_node], &self.theta_names)?;
            let a: Vec<Vec<f64>> = (0..2)
                .map(|r| {
                    (0..2)
                        .map(|c| dot(&j[r], &j[c]) + if r == c { lambda_sq } else { 0.0 })
                        .collect()
                })
                .collect();

            // Solve (J Jᵀ + λ²I) y = e, then map back to joint space: Δθ = Jᵀ y.
            let (l, u, piv) = lu_decompose(&a)?;
            let y = lu_solve(&l, &u, &piv, &e);
            for (col, angle) in theta.iter_mut().enumerate() {
                *angle += j[0][col] * y[0] + j[1][col] * y[1];
            }
        }

        Ok(IkResult {
            angles: theta,
            reached: false,
            history,
        })
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Acceptance criterion 1: a 3-link unit arm reaches a reachable target.
    #[test]
    fn reaches_reachable_target() {
        let mut arm = IkArm::new(&[1.0, 1.0, 1.0], 0.1);
        let target = (1.5, 1.0); // distance ~1.80 from base, well within reach 3
        let result = arm.solve(target, &[0.1, 0.1, 0.1], 1e-6, 200).unwrap();

        assert!(
            result.reached,
            "should converge; final error {}",
            result.history.last().unwrap().error_norm
        );
        let tip = result.history.last().unwrap().tip;
        assert!((tip.0 - target.0).hypot(tip.1 - target.1) < 1e-6);
    }

    // DLS stays well-conditioned from a fully-extended (singular) start, where
    // J's x-row is all zeros and a raw pseudoinverse would blow up: λ²I keeps
    // the 2×2 system solvable, so the arm escapes the singularity and converges.
    #[test]
    fn stable_from_singular_start() {
        let mut arm = IkArm::new(&[1.0, 1.0, 1.0], 0.1);
        // All-zero angles: arm straight along +x, tip at (3,0), J rank-1.
        let result = arm.solve((2.0, 0.5), &[0.0, 0.0, 0.0], 1e-6, 500).unwrap();
        assert!(result.reached);
    }
}
