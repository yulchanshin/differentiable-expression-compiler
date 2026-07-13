//! The engine's error type.
//!
//! [`EngineError`] is returned by the fallible passes (`forward`, `backward`,
//! `jacobian`) for expected failures that depend on runtime input: an unknown
//! variable, division by zero, or a math-domain violation. Programmer bugs
//! (empty graph, bad node index) panic instead.

#[derive(Debug)]
pub enum EngineError {
    UnknownVariable(String), // a Var node's name isn't in the inputs map
    DivByZero,               // div(a, b) with b == 0
    DomainError(String),     // ln(x<=0), pow(neg, fractional), etc.; message says which
}
