//! The expression AST produced by the parser.
//!
//! [`Expr`] is a recursive tree whose `Unary`, `Binary`, and `Call` variants
//! own their children through `Box`. The `Box` is what makes the size finite:
//! it is one pointer, so the enum no longer contains itself by value. A tree
//! node solely owns its subtrees, so `Box` is enough; the compute graph shares
//! subexpressions and needs the arena instead.

use crate::parse::lexer::Token;

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
