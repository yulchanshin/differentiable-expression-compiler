//! Finite-difference oracle — the AD correctness harness (TICKET-104).
//!
//! Integration test: compiles as its own crate and exercises `engine`
//! from the outside via `use engine::...`. The real oracle lands in
//! TICKET-104 once the autodiff core exists; this file just stakes out
//! the location.
