//! The expression AST produced by the parser.
//!
//! [`Expr`] is a recursive tree whose `Unary`/`Binary`/`Call` variants own their
//! children through `Box` (one pointer, so the enum stays a finite size). A tree
//! owns its subtrees outright; the compute graph shares subexpressions and uses
//! the arena instead.

use crate::parse::lexer::Token;

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),
    Unary {
        op: Token,
        child: Box<Expr>,
    },
    Binary {
        op: Token,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Call {
        fn_name: String,
        arg: Box<Expr>,
    },
}
