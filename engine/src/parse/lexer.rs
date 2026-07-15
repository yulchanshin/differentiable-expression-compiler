//! Lexer: turns source text like `sin(x*y) + x^2` into a `Token` stream.
use crate::error::EngineError;
use std::iter::Peekable;
use std::str::Chars;

/// A single lexical token. `Number` carries an `f64`, so the enum can derive
/// `PartialEq` but not `Eq`/`Hash`.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ident(String),
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    Eof,
}

pub struct Lexer<'a> {
    chars: Peekable<Chars<'a>>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Lexer {
            chars: source.chars().peekable(),
        }
    }

    pub fn next_token(&mut self) -> Result<Token, EngineError> {
        // Skip any leading whitespace so the match below sees the first
        // meaningful char (or nothing, at end of input).
        while self.chars.peek().is_some_and(|c| c.is_whitespace()) {
            self.chars.next();
        }

        // Copy the next char out by value, which ends the `peek` borrow so the
        // arms below are free to call `self.chars.next()`. `None` means we've
        // consumed the whole input, which is exactly the Eof token.
        let c = match self.chars.peek() {
            Some(&c) => c,
            None => return Ok(Token::Eof),
        };

        // One match to classify what token starts here. The two multi-char
        // cases hand off to a helper that consumes the whole run; the single
        // char cases consume that one char and return their token.
        match c {
            c if c.is_alphabetic() => Ok(self.read_identifier()),
            c if c.is_ascii_digit() || c == '.' => self.read_number(),
            '+' => {
                self.chars.next();
                Ok(Token::Plus)
            }
            '-' => {
                self.chars.next();
                Ok(Token::Minus)
            }
            '*' => {
                self.chars.next();
                Ok(Token::Star)
            }
            '/' => {
                self.chars.next();
                Ok(Token::Slash)
            }
            '^' => {
                self.chars.next();
                Ok(Token::Caret)
            }
            '(' => {
                self.chars.next();
                Ok(Token::LParen)
            }
            ')' => {
                self.chars.next();
                Ok(Token::RParen)
            }
            ',' => {
                self.chars.next();
                Ok(Token::Comma)
            }
            _ => {
                self.chars.next();
                Err(EngineError::UnexpectedChar(c))
            }
        }
    }

    /// Drive `next_token` to exhaustion, collecting the whole stream into a
    /// `Vec`. The trailing `Eof` is kept in the result so the parser can peek
    /// past the last real token without bounds-checking. The first lexer error
    /// short-circuits via `?`, so a bad source yields `Err`, not a partial Vec.
    pub fn tokenize(&mut self) -> Result<Vec<Token>, EngineError> {
        let mut result: Vec<Token> = Vec::new();
        loop {
            let tok: Token = self.next_token()?;
            let done = tok == Token::Eof;
            result.push(tok);
            if done {
                return Ok(result);
            }
        }
    }

    /// Consume a run of alphabetic chars into an `Ident`. Called only when the
    /// next char is already known to be a letter.
    fn read_identifier(&mut self) -> Token {
        let mut string: String = String::new();
        // peek to decide whether to keep going, next() to actually take the char
        while let Some(&c) = self.chars.peek() {
            if c.is_alphanumeric() {
                string.push(c);
                self.chars.next();
            } else {
                break;
            }
        }
        Token::Ident(string)
    }

    /// Consume a run of digits (with an optional decimal point) into a
    /// `Number`. Called only when the next char is a digit or '.'.
    fn read_number(&mut self) -> Result<Token, EngineError> {
        let mut num: String = String::new();
        while let Some(&n) = self.chars.peek() {
            if n.is_ascii_digit() || n == '.' {
                num.push(n);
                self.chars.next();
            } else {
                break;
            }
        }
        // The loop over-accepts (e.g. "1.2.3"), so parse() is the validation
        // gate: a bad run becomes an Err instead of a panic.
        num.parse::<f64>()
            .map(Token::Number)
            .map_err(|_| EngineError::InvalidNumber(num))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lex `src` end to end via `tokenize`, so these tests drive the same
    /// collection path the parser uses rather than a separate copy of the loop.
    fn lex_all(src: &str) -> Result<Vec<Token>, EngineError> {
        Lexer::new(src).tokenize()
    }

    // The canonical expression from the roadmap: 10 content tokens + Eof.
    #[test]
    fn lexes_sin_xy_plus_x2() {
        let tokens = lex_all("sin(x*y) + x^2").expect("should lex cleanly");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("sin".into()),
                Token::LParen,
                Token::Ident("x".into()),
                Token::Star,
                Token::Ident("y".into()),
                Token::RParen,
                Token::Plus,
                Token::Ident("x".into()),
                Token::Caret,
                Token::Number(2.0),
                Token::Eof,
            ]
        );
    }

    // Each single-char operator and delimiter maps to its own token.
    #[test]
    fn lexes_all_operators() {
        assert_eq!(
            lex_all("+-*/^(),").unwrap(),
            vec![
                Token::Plus,
                Token::Minus,
                Token::Star,
                Token::Slash,
                Token::Caret,
                Token::LParen,
                Token::RParen,
                Token::Comma,
                Token::Eof,
            ]
        );
    }

    // Multi-digit integers and decimals both parse to one Number each.
    #[test]
    fn lexes_multidigit_and_float_numbers() {
        assert_eq!(
            lex_all("42").unwrap(),
            vec![Token::Number(42.0), Token::Eof]
        );
        assert_eq!(
            lex_all("3.14").unwrap(),
            vec![Token::Number(3.14), Token::Eof]
        );
        assert_eq!(
            lex_all("10 + 2.5").unwrap(),
            vec![
                Token::Number(10.0),
                Token::Plus,
                Token::Number(2.5),
                Token::Eof,
            ]
        );
    }

    // An identifier is a letter followed by any run of alphanumerics, so a
    // trailing digit stays part of the same Ident (theta2, not theta then 2).
    #[test]
    fn identifier_allows_trailing_digits() {
        assert_eq!(
            lex_all("theta2").unwrap(),
            vec![Token::Ident("theta2".into()), Token::Eof]
        );
    }

    #[test]
    fn lexes_nested_parens() {
        assert_eq!(
            lex_all("((x))").unwrap(),
            vec![
                Token::LParen,
                Token::LParen,
                Token::Ident("x".into()),
                Token::RParen,
                Token::RParen,
                Token::Eof,
            ]
        );
    }

    // Whitespace (spaces and tabs; leading, trailing, and interior) is skipped.
    #[test]
    fn skips_whitespace_between_tokens() {
        assert_eq!(
            lex_all("  x   +\ty ").unwrap(),
            vec![
                Token::Ident("x".into()),
                Token::Plus,
                Token::Ident("y".into()),
                Token::Eof,
            ]
        );
    }

    // Empty (or all-whitespace) input is just Eof.
    #[test]
    fn empty_input_is_just_eof() {
        assert_eq!(lex_all("").unwrap(), vec![Token::Eof]);
        assert_eq!(lex_all("   ").unwrap(), vec![Token::Eof]);
    }

    // Error case 1: a char that can't start any token.
    #[test]
    fn unexpected_char_errors() {
        let err = lex_all("x @ y").unwrap_err();
        assert!(matches!(err, EngineError::UnexpectedChar('@')));
    }

    // Error case 2: a numeric run that doesn't parse as f64.
    #[test]
    fn malformed_number_errors() {
        let err = lex_all("1.2.3").unwrap_err();
        assert!(matches!(err, EngineError::InvalidNumber(_)));
    }
}
