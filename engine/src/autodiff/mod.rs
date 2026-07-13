//! Automatic differentiation: forward evaluation, the reverse-mode backward
//! pass, Jacobian assembly, and trace emission.

pub mod backward;
pub mod forward;
pub mod jacobian;
pub mod trace;
