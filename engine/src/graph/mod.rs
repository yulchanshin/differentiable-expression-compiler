//! The computation graph: the arena (`Graph`) that owns all nodes, the `Node`
//! and `OpType` data model, structural node keys, formula rendering, and
//! topological-ordering utilities.

pub mod arena;
pub mod display;
pub mod key;
pub mod node;
pub mod topo;
