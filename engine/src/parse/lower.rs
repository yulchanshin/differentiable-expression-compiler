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
//! [`OpType`] carries `f64` payloads (`Const`, `Pow`), and `f64` implements
//! neither `Eq` nor `Hash` because of `NaN` (`NaN != NaN`). A `HashMap` key
//! must be both, so [`NodeKey`] stores floats as their raw `u64` bit patterns.
//! The key type lives in [`crate::graph::key`] because CSE (TICKET-401) keys
//! nodes the same way; lowering uses the order-preserving [`NodeKey::new`].

use std::collections::HashMap;

use crate::error::EngineError;
use crate::graph::arena::Graph;
use crate::graph::key::NodeKey;
use crate::graph::node::{Node, OpType};
use crate::parse::ast::Expr;
use crate::parse::lexer::Token;

/// Memo table for hash-consing: maps a node's structural key to the index of
/// the single node that realizes it.
type Memo = HashMap<NodeKey, usize>;

#[derive(Default)]
pub struct Lowerer {
    memo: Memo,
}

impl Lowerer {
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

/// Lower a whole expression tree into a fresh graph.
///
/// Returns the graph and the index of its root (output) node. Callers use
/// this instead of driving a [`Lowerer`] themselves, so the memo table stays
/// an internal detail. Because interning pushes each node after its inputs,
/// index order is a valid topological order and the root is the last node.
pub fn lower(expr: &Expr) -> Result<(Graph, usize), EngineError> {
    let mut graph = Graph::new();
    let root = Lowerer::default().lower(expr, &mut graph)?;
    Ok((graph, root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::compile;
    use std::collections::HashMap;

    #[test]
    fn shared_subexpression_is_one_node() {
        // x*y + x*y: hash-consing must collapse the repeated product so the
        // graph is exactly {x, y, x*y, +} -- four nodes, not seven.
        let (g, root) = compile("x*y + x*y").unwrap();
        assert_eq!(g.nodes.len(), 4);

        // The root Add's two inputs are the *same* x*y node.
        let inputs = &g.nodes[root].inputs;
        assert_eq!(inputs[0], inputs[1]);
    }

    #[test]
    fn repeated_variable_is_one_node() {
        // x appears twice (in x*y and x^2) but must be a single shared node.
        let (g, _) = compile("sin(x*y) + x^2").unwrap();
        let x_nodes = g
            .nodes
            .iter()
            .filter(|n| matches!(&n.op, OpType::Var(name) if name == "x"))
            .count();
        assert_eq!(x_nodes, 1);
    }

    #[test]
    fn matches_hand_built_graph() {
        // Acceptance criterion: parsing + lowering sin(x*y)+x^2 produces the
        // same graph (node count and forward value) as the hand-built version
        // from the forward-pass tests.
        let (mut g, _) = compile("sin(x*y) + x^2").unwrap();
        assert_eq!(g.nodes.len(), 6); // x, y, x*y, sin, x^2, +

        let inputs = HashMap::from([("x".to_string(), 1.5), ("y".to_string(), 2.0)]);
        let result = g.forward(&inputs).expect("forward should succeed");
        let expected = (1.5_f64 * 2.0).sin() + 1.5_f64.powi(2);
        assert!((result - expected).abs() < 1e-9);
    }

    #[test]
    fn unknown_function_is_an_error() {
        assert!(matches!(
            compile("sqrt(x)"),
            Err(EngineError::UnexpectedToken { .. })
        ));
    }
}
