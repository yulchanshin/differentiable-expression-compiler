//! Structural fingerprints for graph nodes: [`OpKey`] and [`NodeKey`].
//!
//! Two subexpressions are "the same" when they have the same op and the same
//! inputs. To detect that with a [`HashMap`](std::collections::HashMap) we need
//! a key that is both `Eq` and `Hash`. [`OpType`] can't serve directly: its
//! `Const`/`Pow` payloads are `f64`, which implements neither trait
//! (`NaN != NaN`). [`OpKey`] mirrors [`OpType`] but stores those floats as their
//! raw `u64` bit patterns via [`f64::to_bits`], giving exact-bit deduplication:
//! two `2.0`s collapse, and we make no attempt to prove `0.1 + 0.2 == 0.3`.
//!
//! Two constructors exist because two passes want slightly different notions of
//! "same". [`NodeKey::new`] preserves operand order and backs lowering's
//! hash-consing, where the AST already fixes operand order. [`NodeKey::canonical`]
//! additionally sorts the operands of commutative ops so `a*b` and `b*a` key
//! identically, which is what common-subexpression elimination needs.

use crate::graph::node::OpType;

/// The op portion of a [`NodeKey`], with every `f64` payload replaced by its
/// `to_bits()` representation so the whole thing can derive `Eq` + `Hash`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum OpKey {
    Var(String),
    Const(u64), // f64::to_bits
    Add,
    Sub,
    Div,
    Mul,
    Neg,
    Pow(u64), // f64::to_bits of the exponent
    Sin,
    Cos,
    Exp,
    Ln,
}

impl OpKey {
    /// The bit-exact key mirror of an [`OpType`].
    fn from_op(op: &OpType) -> Self {
        match op {
            OpType::Var(name) => OpKey::Var(name.clone()),
            OpType::Const(v) => OpKey::Const(v.to_bits()),
            OpType::Add => OpKey::Add,
            OpType::Sub => OpKey::Sub,
            OpType::Div => OpKey::Div,
            OpType::Mul => OpKey::Mul,
            OpType::Neg => OpKey::Neg,
            OpType::Pow(e) => OpKey::Pow(e.to_bits()),
            OpType::Sin => OpKey::Sin,
            OpType::Cos => OpKey::Cos,
            OpType::Exp => OpKey::Exp,
            OpType::Ln => OpKey::Ln,
        }
    }

    /// Whether operand order is irrelevant to the node's value. Only `Add` and
    /// `Mul` commute; `Sub`/`Div` do not, and every remaining op is unary.
    pub fn is_commutative(&self) -> bool {
        matches!(self, OpKey::Add | OpKey::Mul)
    }
}

/// A structural fingerprint of a node: its op plus the indices of its inputs.
/// Two subexpressions that build the same key are the same node.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct NodeKey {
    pub op: OpKey,
    pub inputs: Vec<usize>,
}

impl NodeKey {
    /// Structural key with operand order preserved. Used by lowering, where the
    /// AST already fixes operand order.
    pub fn new(op: &OpType, inputs: &[usize]) -> Self {
        NodeKey {
            op: OpKey::from_op(op),
            inputs: inputs.to_vec(),
        }
    }

    /// Structural key with commutative operands canonicalized: the inputs of an
    /// `Add`/`Mul` are sorted so `a*b` and `b*a` produce the same key. Used by
    /// CSE to merge nodes that differ only in operand order.
    pub fn canonical(op: &OpType, inputs: &[usize]) -> Self {
        let mut key = NodeKey::new(op, inputs);
        if key.op.is_commutative() {
            key.inputs.sort_unstable();
        }
        key
    }
}
