pub mod ast;
pub mod lexer;
pub mod lower;
pub mod parser;

use crate::error::EngineError;
use crate::parse::ast::Expr;
use crate::parse::lexer::{Lexer, Token};
use crate::parse::parser::Parser;

pub fn parse(src: &str) -> Result<Expr, EngineError> {
    let tokens: Vec<Token> = Lexer::new(src).tokenize()?;
    Parser::new(tokens).parse()
}
