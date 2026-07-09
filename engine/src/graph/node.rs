//! The `Node` struct and `OpType` enum — the graph's data model.

pub struct Node {
    pub op: OpType,
    pub inputs: Vec<usize>,
    pub value: f64,
    pub adjoint: f64,
}

pub enum OpType {
    Var(String),
    Const(f64),
    Add,
    Sub,
    Div,
    Mul,
    Neg,
    Pow(f64),
    Sin,
    Cos,
    Exp,
    Ln,
}
