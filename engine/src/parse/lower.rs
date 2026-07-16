//! Lowering the AST into the arena graph with hash-consing.
//!
//! The parser produces an [`Expr`] tree where every subexpression is owned
//! exactly once. The compute graph is different: identical subexpressions
//! should be a single *shared* node so that work done once is reused
//! everywhere. `x*y + x*y` should build one `x*y` node, not two.
//!
//! We get that sharing for free with **hash-consing**: before creating a
//! node we look it up in a [`HashMap`] keyed by its structure `(op, inputs)`.
//! Because inputs are indices into already-deduplicated nodes, structural
//! equality of two subexpressions reduces to equality of their keys.
//!
//! ## Why [`NodeKey`] and not [`OpType`] as the key
//!
//! [`OpType`] carries `f64` payloads (`Const`, `Pow`), and `f64` implements
//! neither `Eq` nor `Hash` because of `NaN` (`NaN != NaN`). A `HashMap` key
//! must be both. So the key stores floats as their raw `u64` bit patterns via
//! [`f64::to_bits`], which gives exact-bit deduplication: two `2.0`s collapse,
//! and we make no attempt to prove `0.1 + 0.2 == 0.3`. Keeping [`NodeKey`]
//! separate also stops a lowering-only concern from constraining the graph's
//! runtime data model.

use std::collections::HashMap;

use crate::error::EngineError;
use crate::graph::arena::Graph;
use crate::graph::node::{Node, OpType};
use crate::parse::ast::Expr;
use crate::parse::lexer::Token;

/// The op portion of a [`NodeKey`], with every `f64` payload replaced by its
/// `to_bits()` representation so the whole thing can derive `Eq` + `Hash`.
#[derive(Clone, PartialEq, Eq, Hash)]
enum OpKey {
    Var(String),
    Const(u64), // f64::to_bits
    Add,
    Sub,
    Div,
    Mul,
    Neg,
    Pow(u64), // f64::to_bits of the exponent
    Sin,
    Cos,
    Exp,
    Ln,
}

/// A structural fingerprint of a node: its op plus the indices of its inputs.
/// Two subexpressions that build the same key are the same node.
#[derive(Clone, PartialEq, Eq, Hash)]
struct NodeKey {
    op: OpKey,
    inputs: Vec<usize>,
}

impl NodeKey {
    /// Build a key from a node's op and its (already-deduplicated) inputs.
    fn new(op: &OpType, inputs: &[usize]) -> Self {
        let op = match op {
            OpType::Var(name) => OpKey::Var(name.clone()),
            OpType::Const(v) => OpKey::Const(v.to_bits()),
            OpType::Add => OpKey::Add,
            OpType::Sub => OpKey::Sub,
            OpType::Div => OpKey::Div,
            OpType::Mul => OpKey::Mul,
            OpType::Neg => OpKey::Neg,
            OpType::Pow(e) => OpKey::Pow(e.to_bits()),
            OpType::Sin => OpKey::Sin,
            OpType::Cos => OpKey::Cos,
            OpType::Exp => OpKey::Exp,
            OpType::Ln => OpKey::Ln,
        };
        NodeKey {
            op,
            inputs: inputs.to_vec(),
        }
    }
}
/// Memo table for hash-consing: maps a node's structural key to the index of
/// the single node that realizes it.
type Memo = HashMap<NodeKey, usize>;

#[derive(Default)]
pub struct Lowerer {
    memo: Memo,
}

impl Lowerer {
    pub fn new() -> Self {
        Lowerer {
            memo: HashMap::new(),
        }
    }

    pub fn lower(&mut self, expr: &Expr, graph: &mut Graph) -> Result<usize, EngineError> {
        match expr {
            Expr::Num(n) => Ok(self.intern(graph, OpType::Const(*n), vec![])),
            Expr::Var(name) => Ok(self.intern(graph, OpType::Var(name.clone()), vec![])),
            Expr::Binary { op, left, right } => {
                // `^` is special: Pow carries a *constant* exponent, so the
                // right side must be a numeric literal and is NOT lowered as a
                // child.
                if let Token::Caret = op {
                    let base = self.lower(left, graph)?;
                    let exp = match right.as_ref() {
                        Expr::Num(e) => *e,
                        _ => {
                            return Err(EngineError::UnexpectedToken {
                                expected: "a numeric literal exponent".to_string(),
                                found: format!("{:?}", right),
                            });
                        }
                    };
                    return Ok(self.intern(graph, OpType::Pow(exp), vec![base]));
                }

                // Everything else lowers both children, then interns.
                let l = self.lower(left, graph)?;
                let r = self.lower(right, graph)?;
                let optype = match op {
                    Token::Plus => OpType::Add,
                    Token::Minus => OpType::Sub,
                    Token::Star => OpType::Mul,
                    Token::Slash => OpType::Div,
                    other => {
                        return Err(EngineError::UnexpectedToken {
                            expected: "a binary operator".to_string(),
                            found: format!("{:?}", other),
                        });
                    }
                };
                Ok(self.intern(graph, optype, vec![l, r]))
            }
            Expr::Unary { op, child } => {
                let child_node: usize = self.lower(child, graph)?;
                let optype = match op {
                    Token::Minus => OpType::Neg,
                    other => {
                        return Err(EngineError::UnexpectedToken {
                            expected: "a unary operator".to_string(),
                            found: format!("{:?}", other),
                        });
                    }
                };
                Ok(self.intern(graph, optype, vec![child_node]))
            }
            // TODO(you): match fn_name ("sin"/"cos"/"exp"/"ln"), error otherwise.
            Expr::Call { fn_name, arg } => {
                let child_node: usize = self.lower(arg, graph)?;
                let optype = match fn_name.as_str() {
                    "sin" => OpType::Sin,
                    "cos" => OpType::Cos,
                    "ln" => OpType::Ln,
                    "exp" => OpType::Exp,
                    other => {
                        return Err(EngineError::UnexpectedToken {
                            expected: "a known function name".to_string(),
                            found: other.to_string(),
                        });
                    }
                };

                Ok(self.intern(graph, optype, vec![child_node]))
            }
        }
    }

    /// The shared create-or-reuse step every arm funnels through.
    fn intern(&mut self, graph: &mut Graph, op: OpType, inputs: Vec<usize>) -> usize {
        let key = NodeKey::new(&op, &inputs);
        if let Some(&idx) = self.memo.get(&key) {
            return idx;
        }
        let idx = graph.push(Node {
            op,
            inputs,
            value: 0.0,
            adjoint: 0.0,
        });
        self.memo.insert(key, idx);
        idx
    }
}
