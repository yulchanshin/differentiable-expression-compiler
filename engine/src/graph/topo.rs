//! Topological ordering utilities.
//!
//! The arena relies on build-order being a valid topological order: the builder
//! helpers always push a node's inputs before the node itself, so a node's
//! inputs sit at lower indices and plain index order suffices for the forward
//! pass (reverse index order for the backward pass). This module is reserved
//! for an explicit topological sort (e.g. Kahn's algorithm) should that
//! invariant ever need to be enforced independently of build order.
