#[derive(Debug)]
pub enum EngineError {
    UnknownVariable(String), // a Var node's name isn't in the inputs map
    DivByZero,               // div(a, b) with b == 0
    DomainError(String),     // ln(x<=0), pow(neg, fractional), etc. — message says which
}
