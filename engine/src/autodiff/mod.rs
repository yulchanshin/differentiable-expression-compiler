//! Automatic differentiation: forward evaluation, the reverse-mode backward
//! pass, Jacobian assembly, trace emission, and symbolic (graph-to-graph)
//! differentiation.

pub mod backward;
pub mod forward;
pub mod jacobian;
pub mod symbolic;
pub mod trace;
