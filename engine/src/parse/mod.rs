pub mod ast;
pub mod lexer;
pub mod lower;
pub mod parser;

use crate::error::EngineError;
use crate::graph::arena::Graph;
use crate::parse::ast::Expr;
use crate::parse::lexer::{Lexer, Token};
use crate::parse::lower::lower;
use crate::parse::parser::Parser;

/// Step 1: turn source text into the full token stream (trailing `Eof`
/// included).
pub fn lex(src: &str) -> Result<Vec<Token>, EngineError> {
    Lexer::new(src).tokenize()
}

/// Step 2: turn a token stream into an expression tree.
pub fn parse(tokens: Vec<Token>) -> Result<Expr, EngineError> {
    Parser::new(tokens).parse()
}

/// The full front end: `lex`, then `parse`, then `lower`.
///
/// Returns the graph and the index of its root (output) node. This is the one
/// entry point callers need to go from source text to a runnable graph; the
/// three stages remain public for exercising each layer on its own.
pub fn compile(src: &str) -> Result<(Graph, usize), EngineError> {
    lower(&parse(lex(src)?)?)
}
