//! Per-op local derivative rules.
//!
//! Reserved to factor each op's forward rule and its derivative into one place
//! so they can't drift. They currently live inline in `autodiff::backward` (the
//! optional TICKET-200 refactor).
