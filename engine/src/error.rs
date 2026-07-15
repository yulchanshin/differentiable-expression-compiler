//! The engine's error type.
//!
//! [`EngineError`] is returned by the fallible passes (`forward`, `backward`,
//! `jacobian`) for expected failures that depend on runtime input: an unknown
//! variable, division by zero, or a math-domain violation. The lexer also
//! returns it for source text it can't tokenize. Programmer bugs (empty graph,
//! bad node index) panic instead.

#[derive(Debug)]
pub enum EngineError {
    UnknownVariable(String), // a Var node's name isn't in the inputs map
    DivByZero,               // div(a, b) with b == 0
    DomainError(String),     // ln(x<=0), pow(neg, fractional), etc.; message says which
    UnexpectedChar(char),    // lexer hit a char that can't start any token
    InvalidNumber(String),   // a numeric run that doesn't parse as f64, e.g. "1.2.3"
    UnexpectedToken {        // parser saw a token the grammar doesn't allow here
        expected: String,    // human description of what was wanted, e.g. "`)`"
        found: String,       // the token actually seen, e.g. "Eof"
    },
}
