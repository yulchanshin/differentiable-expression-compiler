//! Pratt (top-down operator-precedence) parser: turns the lexer's token stream
//! into an [`Expr`] tree.
//!
//! `parse_expr` is the core. It seeds a left operand from `parse_atom`, then
//! folds in infix operators for as long as they bind at least as tightly as the
//! caller's `min_bp`. Precedence and associativity live entirely in the
//! `(left_bp, right_bp)` pairs of `infix_binding_power`, so the loop stays small
//! and those tables are the only thing to touch when adding an operator.

use crate::error::EngineError;
use crate::parse::ast::Expr;
use crate::parse::lexer::Token;

pub struct Parser {
    pub tokens: Vec<Token>,
    pos: usize, //index of current token
}

impl Parser {
    pub fn parse(&mut self) -> Result<Expr, EngineError> {
        let expr = self.parse_expr(0)?;
        self.expect(Token::Eof)?; // nothing may follow the expression
        Ok(expr)
    }
    /// Build a parser over an already-lexed token stream. Expects the trailing
    /// `Eof` from `Lexer::tokenize` to be present, which is what lets `peek`
    /// stay in bounds once the real tokens run out. The cursor starts at 0.
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    /// Borrow the current token without consuming it. Safe to call at the end
    /// of input because the stream always ends in `Eof`, so the cursor parks on
    /// that final token instead of running past the end of the vector.
    pub fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    /// Consume the current token and step the cursor forward one. Clones the
    /// token out rather than moving it, since the `Vec` keeps owning its slot
    /// and `Token` is not `Copy`.
    pub fn advance(&mut self) -> Token {
        let tok: Token = self.tokens[self.pos].clone();
        self.pos += 1;
        tok
    }

    /// Consume the current token if it equals `expected`, otherwise return a
    /// descriptive error. Used for tokens the grammar requires in a fixed spot,
    /// like the `)` that must close a `(` group. On success the cursor advances
    /// past the matched token; on failure the cursor is left where it was.
    pub fn expect(&mut self, expected: Token) -> Result<(), EngineError> {
        if self.peek() == &expected {
            self.advance();
            Ok(())
        } else {
            Err(EngineError::UnexpectedToken {
                expected: format!("{expected:?}"),
                found: format!("{:?}", self.peek()),
            })
        }
    }

    /// Parse a single primary expression: the smallest thing that can *start*
    /// an expression on its own. `advance` up front hands us an owned `Token`,
    /// which ends the borrow of `self` so the arms are free to call back into
    /// `parse_expr`/`expect`, and gives the leaf arms owned values with no
    /// clone. Handles numbers, variables, function calls, parenthesized
    /// groups, and prefix `-`.
    pub fn parse_atom(&mut self) -> Result<Expr, EngineError> {
        match self.advance() {
            Token::Number(n) => Ok(Expr::Num(n)),
            Token::Ident(name) => {
                if matches!(self.peek(), Token::LParen) {
                    self.advance();
                    let arg: Expr = self.parse_expr(0)?;
                    self.expect(Token::RParen)?;
                    Ok(Expr::Call {
                        fn_name: name,
                        arg: Box::new(arg),
                    })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            // `(` resets precedence: parse a fresh expression, then require the
            // matching `)`. The parens leave no node; we return the inner tree.
            Token::LParen => {
                let inner: Expr = self.parse_expr(0)?;
                self.expect(Token::RParen)?;
                Ok(inner)
            }
            Token::Minus => Ok(Expr::Unary {
                op: Token::Minus,
                child: Box::new(self.parse_expr(5)?),
            }),
            other => Err(EngineError::UnexpectedToken {
                expected: "a number, variable, or '('".into(),
                found: format!("{other:?}"),
            }),
        }
    }

    /// The Pratt core. Parse an expression, only absorbing operators that bind
    /// at least as tightly as `min_bp`. The caller sets `min_bp` to fence off
    /// operators it wants to keep for itself, which is how precedence and
    /// associativity fall out of the recursion.
    pub fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, EngineError> {
        // Step 1 (seed): every expression starts with one primary operand.
        let mut lhs: Expr = self.parse_atom()?;

        loop {
            // Step 2: peek the candidate operator and look up its power. A
            // token that isn't an infix operator (Eof, `)`, ...) ends the loop.
            let op: Token = self.peek().clone();
            let (left_bp, right_bp) = match infix_binding_power(&op) {
                Some(bp) => bp,
                None => break,
            };

            // Step 3: if it binds looser than the caller allows, it is not ours
            // to take. Stop and let the caller (with a lower min_bp) absorb it.
            if left_bp < min_bp {
                break;
            }

            // Step 4: commit to the operator, then parse its right operand with
            // right_bp. Right-associativity of `^` is encoded purely in how
            // right_bp compares to left_bp (see the helper below).
            self.advance();
            let rhs: Expr = self.parse_expr(right_bp)?;

            // Step 5: fold operator + both operands into a node, which becomes
            // the new left operand for any further operators in this loop.
            lhs = Expr::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
            };
        }

        Ok(lhs)
    }
}

/// Binding power of an infix operator, as `(left_bp, right_bp)`. Higher binds
/// tighter. Left-associative operators give `right > left` (so a same-power
/// operator to the right is fenced out, grouping leftward); `^` is
/// right-associative, so it flips to `right < left`. Returns `None` for any
/// token that is not an infix operator, which is the loop's stop signal.
fn infix_binding_power(tok: &Token) -> Option<(u8, u8)> {
    match tok {
        Token::Plus | Token::Minus => Some((1, 2)),
        Token::Star | Token::Slash => Some((3, 4)),
        Token::Caret => Some((7, 6)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lex then fully parse a valid expression (through `parse`, so the
    /// trailing-`Eof` check runs too), unwrapping both steps.
    fn parse(src: &str) -> Expr {
        let tokens = crate::parse::lexer::Lexer::new(src).tokenize().unwrap();
        Parser::new(tokens).parse().unwrap()
    }

    /// Lex then parse source expected to be invalid, returning the error so a
    /// test can assert on its kind. Panics if the input unexpectedly parses.
    fn parse_err(src: &str) -> EngineError {
        let tokens = crate::parse::lexer::Lexer::new(src).tokenize().unwrap();
        Parser::new(tokens).parse().unwrap_err()
    }

    // Small tree builders so the expected ASTs read close to the source.
    fn var(name: &str) -> Expr {
        Expr::Var(name.into())
    }
    fn num(n: f64) -> Expr {
        Expr::Num(n)
    }
    fn bin(op: Token, left: Expr, right: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }
    fn unary(op: Token, child: Expr) -> Expr {
        Expr::Unary {
            op,
            child: Box::new(child),
        }
    }
    fn call(fn_name: &str, arg: Expr) -> Expr {
        Expr::Call {
            fn_name: fn_name.into(),
            arg: Box::new(arg),
        }
    }

    // `a + b*c` groups the multiply first: a + (b*c).
    #[test]
    fn multiply_binds_tighter_than_add() {
        assert_eq!(
            parse("a + b*c"),
            bin(Token::Plus, var("a"), bin(Token::Star, var("b"), var("c"))),
        );
    }

    // `^` is right-associative: x^2^3 parses as x^(2^3).
    #[test]
    fn caret_is_right_associative() {
        assert_eq!(
            parse("x^2^3"),
            bin(
                Token::Caret,
                var("x"),
                bin(Token::Caret, num(2.0), num(3.0)),
            ),
        );
    }

    // `-` is left-associative: a-b-c parses as (a-b)-c.
    #[test]
    fn minus_is_left_associative() {
        assert_eq!(
            parse("a-b-c"),
            bin(
                Token::Minus,
                bin(Token::Minus, var("a"), var("b")),
                var("c"),
            ),
        );
    }

    // Parens override precedence and leave no node of their own.
    #[test]
    fn parens_regroup_and_leave_no_node() {
        assert_eq!(
            parse("(a+b)*c"),
            bin(Token::Star, bin(Token::Plus, var("a"), var("b")), var("c")),
        );
    }

    // Prefix `-` binds looser than `^`: -x^2 parses as -(x^2), not (-x)^2.
    #[test]
    fn prefix_minus_binds_looser_than_caret() {
        assert_eq!(
            parse("-x^2"),
            unary(Token::Minus, bin(Token::Caret, var("x"), num(2.0))),
        );
    }

    // A function call wraps a parenthesized argument expression.
    #[test]
    fn call_parses_name_and_argument() {
        assert_eq!(
            parse("sin(x*y)"),
            call("sin", bin(Token::Star, var("x"), var("y"))),
        );
    }

    // Error case 1: a `(` group that is never closed. The inner parse_expr
    // stops on Eof, so expect(RParen) fails instead of running off the end.
    #[test]
    fn unclosed_paren_errors() {
        assert!(matches!(
            parse_err("(a+b"),
            EngineError::UnexpectedToken { .. }
        ));
    }

    // Error case 2: trailing tokens after a complete expression. parse_expr
    // returns `1`, then parse's expect(Eof) rejects the leftover `2`.
    #[test]
    fn trailing_tokens_error() {
        assert!(matches!(
            parse_err("1 2"),
            EngineError::UnexpectedToken { .. }
        ));
    }

    // Error case 3: an operator with no right operand. parse_atom is asked for
    // a primary but finds Eof, which can't start an expression.
    #[test]
    fn missing_operand_errors() {
        assert!(matches!(
            parse_err("a *"),
            EngineError::UnexpectedToken { .. }
        ));
    }

    // Error case 4: a token that cannot begin an expression at all.
    #[test]
    fn expression_cannot_start_with_rparen() {
        assert!(matches!(
            parse_err(")"),
            EngineError::UnexpectedToken { .. }
        ));
    }
}
