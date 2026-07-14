//lexer.rs
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
        todo!("loop while peek() is a digit or '.', parse::<f64>(), return Token::Number")
        let mut num: String = String::new();
        while let Some(&n) = self.chars.peek(){
            if n.is_ascii_digit() || n == '.' {
                num.push(n);
                self.chars.next();
            } else {
                break;
            }

        }
        Ok(Token::Number(num.parse().unwrap())) }
}
