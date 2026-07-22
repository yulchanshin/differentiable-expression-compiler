//! Topological ordering utilities.
//!
//! Build order is already a valid topological order (inputs are pushed before
//! their node), so plain index order drives the forward pass and reverse index
//! order the backward pass. Reserved for an explicit sort (e.g. Kahn's) should
//! that ever need enforcing independently of build order.
