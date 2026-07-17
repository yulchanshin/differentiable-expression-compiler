//! Shared scalar evaluation of a single operator node.
//!
//! Both the forward pass and constant folding turn one operator node plus its
//! inputs' values into a scalar, applying the same domain checks. Keeping that
//! logic here means the two callers can't drift: `forward` propagates the
//! `Err`, while `const_fold` reads it as "leave this node unfolded".

use crate::error::EngineError;
use crate::graph::node::OpType;

/// Evaluate a single operator node from its inputs' already-computed values.
///
/// `vals` holds the values of the node's inputs in order (two for a binary op,
/// one for a unary op). Returns the same [`EngineError`]s the forward pass
/// surfaces for domain failures: division by zero, `ln` of a non-positive
/// value, and `pow` of a negative base to a fractional exponent.
///
/// # Panics
/// Panics if called on a leaf (`Const`/`Var`): those carry no operands and are
/// handled by callers directly.
pub fn eval_op(op: &OpType, vals: &[f64]) -> Result<f64, EngineError> {
    let value = match op {
        OpType::Add => vals[0] + vals[1],
        OpType::Sub => vals[0] - vals[1],
        OpType::Mul => vals[0] * vals[1],
        OpType::Div => {
            if vals[1] == 0.0 {
                return Err(EngineError::DivByZero);
            }
            vals[0] / vals[1]
        }
        OpType::Neg => -vals[0],
        OpType::Pow(n) => {
            if vals[0] < 0.0 && n.fract() != 0.0 {
                return Err(EngineError::DomainError(format!(
                    "pow with negative base {} and non-integer exponent {n}",
                    vals[0]
                )));
            }
            vals[0].powf(*n)
        }
        OpType::Sin => vals[0].sin(),
        OpType::Cos => vals[0].cos(),
        OpType::Exp => vals[0].exp(),
        OpType::Ln => {
            if vals[0] <= 0.0 {
                return Err(EngineError::DomainError(format!(
                    "ln requires x > 0, but got {}",
                    vals[0]
                )));
            }
            vals[0].ln()
        }
        OpType::Const(_) | OpType::Var(_) => {
            unreachable!("eval_op called on a leaf node ({op:?})")
        }
    };
    Ok(value)
}
