//! The computation graph: the arena (`Graph`) that owns all nodes, the `Node`
//! and `OpType` data model, and topological-ordering utilities.

pub mod arena;
pub mod node;
pub mod topo;
