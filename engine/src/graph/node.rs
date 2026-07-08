//node.rs
//!description incoming

pub struct Node {
    pub op: OpType,
    pub inputs: Vec<usize>,
    pub value: f64,
    pub adjoint: f64,
}

pub enum OpType {
    Var,
    Mul,
}
