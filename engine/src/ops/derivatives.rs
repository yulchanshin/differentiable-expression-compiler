//! Per-op local derivative rules.
//!
//! The §4.3 local derivatives currently live inline in the backward pass
//! (`autodiff::backward`), kept adjacent to nothing else. This module is
//! reserved for factoring each op's forward rule and its derivative into one
//! place so the two can't drift (the optional refactor noted in TICKET-200).
