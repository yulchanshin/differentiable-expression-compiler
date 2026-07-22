//! Per-op local derivative rules.
//!
//! Reserved to factor each op's derivative into one place so the two consumers
//! can't drift. The rules currently live in two independent sites: the
//! reverse-mode partials in `autodiff::backward`, and the graph-to-graph
//! rewrites in `autodiff::symbolic`. Adding an `OpType` means updating both (as
//! well as forward eval in `ops::eval`). The optional TICKET-200 refactor.
