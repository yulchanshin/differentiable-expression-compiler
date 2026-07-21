//! Render a subgraph back into a readable infix formula.
//!
//! [`Graph::to_expr_string`] walks the subgraph rooted at a node and emits
//! surface syntax like `cos(x * y) * y`, inserting parentheses only where
//! operator precedence requires them. This is what makes symbolic
//! differentiation legible (print `f` and `df/dx` as formulas) and is the
//! formula artifact a symbolic-diff frontend renders.
//!
//! ## Precedence and associativity
//! The precedence levels below mirror the parser's binding powers
//! (`parse::parser::infix_binding_power`), so the output re-parses with the same
//! grouping. Each node reports the precedence of its top operator; a child is
//! parenthesized when it binds *looser* than its parent, plus the
//! associativity tie-breaks: for left-associative `+ - * /` the right operand
//! is wrapped on a tie (so `a - (b - c)` stays grouped), and a `^` base is
//! wrapped on a tie (so `(x ^ 2) ^ 3` stays grouped, since `^` is
//! right-associative).
//!
//! ## Not a perfect round-trip
//! A formula prints for readability, not guaranteed re-lowering. `^` carries a
//! *constant* exponent and the lexer has no negative number literal, so a
//! `Pow` with a negative or fractional exponent (which derivatives routinely
//! produce, e.g. `x ^ -0.5`) prints legibly but does not lower back through the
//! front end. Formulas built only from non-negative literal exponents round-trip
//! by value.

use crate::graph::arena::Graph;
use crate::graph::node::OpType;

// Precedence levels, higher binds tighter. Mirrors the parser's binding powers.
const P_ADD: u8 = 1; // + and -
const P_MUL: u8 = 2; // * and /
const P_NEG: u8 = 3; // prefix -
const P_POW: u8 = 4; // ^
const P_ATOM: u8 = 5; // variables, constants, function calls

impl Graph {
    /// Render the subgraph rooted at `root` as an infix formula string.
    pub fn to_expr_string(&self, root: usize) -> String {
        self.fmt_node(root).0
    }

    /// Format one node, returning its rendered form and the precedence of its
    /// top operator (so the caller can decide whether to parenthesize it).
    fn fmt_node(&self, node: usize) -> (String, u8) {
        match &self.nodes[node].op {
            OpType::Var(name) => (name.clone(), P_ATOM),
            OpType::Const(c) => (fmt_num(*c), P_ATOM),

            OpType::Add => (self.fmt_binary(node, "+", P_ADD), P_ADD),
            OpType::Sub => (self.fmt_binary(node, "-", P_ADD), P_ADD),
            OpType::Mul => (self.fmt_binary(node, "*", P_MUL), P_MUL),
            OpType::Div => (self.fmt_binary(node, "/", P_MUL), P_MUL),

            OpType::Neg => {
                let (s, p) = self.fmt_node(self.nodes[node].inputs[0]);
                // Prefix `-` binds tighter than `* /` but looser than `^`, so an
                // operand looser than P_NEG needs parens: `-(x * y)`, but `-x ^ 2`.
                let inner = if p < P_NEG { format!("({s})") } else { s };
                (format!("-{inner}"), P_NEG)
            }
            OpType::Pow(k) => {
                let (s, p) = self.fmt_node(self.nodes[node].inputs[0]);
                // `^` is right-associative, so its base is wrapped on a tie too.
                let base = if p <= P_POW { format!("({s})") } else { s };
                (format!("{base} ^ {}", fmt_num(*k)), P_POW)
            }

            OpType::Sin => (self.fmt_call(node, "sin"), P_ATOM),
            OpType::Cos => (self.fmt_call(node, "cos"), P_ATOM),
            OpType::Exp => (self.fmt_call(node, "exp"), P_ATOM),
            OpType::Ln => (self.fmt_call(node, "ln"), P_ATOM),
        }
    }

    /// Format a binary op `left OP right`, parenthesizing each operand by
    /// precedence. Both operators here are left-associative, so the right
    /// operand is wrapped on a precedence tie while the left is not.
    fn fmt_binary(&self, node: usize, op: &str, prec: u8) -> String {
        let (ls, lp) = self.fmt_node(self.nodes[node].inputs[0]);
        let (rs, rp) = self.fmt_node(self.nodes[node].inputs[1]);
        let left = if lp < prec { format!("({ls})") } else { ls };
        let right = if rp <= prec { format!("({rs})") } else { rs };
        format!("{left} {op} {right}")
    }

    /// Format a unary function call `name(arg)`. The call's own parens fully
    /// delimit the argument, so it never needs extra wrapping.
    fn fmt_call(&self, node: usize, name: &str) -> String {
        let (s, _) = self.fmt_node(self.nodes[node].inputs[0]);
        format!("{name}({s})")
    }
}

/// Format a constant/exponent. Rust's default float formatting drops a trailing
/// `.0`, so integer-valued numbers print as `2` (which re-lexes as a number).
fn fmt_num(x: f64) -> String {
    format!("{x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::compile;
    use rand::RngExt;
    use std::collections::HashMap;

    #[test]
    fn atoms() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        assert_eq!(g.to_expr_string(x), "x");
        let c = g.constant(2.5);
        assert_eq!(g.to_expr_string(c), "2.5");
        let n = g.constant(2.0);
        assert_eq!(g.to_expr_string(n), "2"); // trailing .0 dropped
    }

    #[test]
    fn precedence_no_parens_when_unneeded() {
        // x + y * z groups the multiply, no parens needed.
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let z = g.var("z".into());
        let yz = g.mul(y, z);
        let root = g.add(x, yz);
        assert_eq!(g.to_expr_string(root), "x + y * z");
    }

    #[test]
    fn precedence_parenthesizes_looser_child() {
        // (x + y) * z: the sum binds looser than the product, so it is wrapped.
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let z = g.var("z".into());
        let xy = g.add(x, y);
        let root = g.mul(xy, z);
        assert_eq!(g.to_expr_string(root), "(x + y) * z");
    }

    #[test]
    fn right_associativity_tie_is_wrapped() {
        // x - (y - z): a right-side subtraction at equal precedence needs parens,
        // otherwise it would re-read as (x - y) - z.
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let z = g.var("z".into());
        let yz = g.sub(y, z);
        let root = g.sub(x, yz);
        assert_eq!(g.to_expr_string(root), "x - (y - z)");
    }

    #[test]
    fn negation_and_pow() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());

        // -(x * y): prefix minus over a looser product is wrapped.
        let xy = g.mul(x, y);
        let neg = g.neg(xy);
        assert_eq!(g.to_expr_string(neg), "-(x * y)");

        // -x ^ 2: pow binds tighter than prefix minus, so no parens.
        let pow = g.pow(x, 2.0);
        let neg_pow = g.neg(pow);
        assert_eq!(g.to_expr_string(neg_pow), "-x ^ 2");

        // (x + y) ^ 2: a looser base is wrapped.
        let sum = g.add(x, y);
        let pow_sum = g.pow(sum, 2.0);
        assert_eq!(g.to_expr_string(pow_sum), "(x + y) ^ 2");
    }

    #[test]
    fn function_call() {
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let xy = g.mul(x, y);
        let s = g.sin(xy);
        assert_eq!(g.to_expr_string(s), "sin(x * y)");
    }

    #[test]
    fn derivative_reads_as_a_formula() {
        // f = sin(x * y);  df/dx = cos(x * y) * y  (before any simplification the
        // chain rule also multiplies by d(x*y)/dx = 1*y + x*0, so this exercises
        // the printer on a real, unsimplified derivative subgraph).
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let xy = g.mul(x, y);
        let f = g.sin(xy);
        let d = g.diff(f, "x");
        // cos(x * y) * (1 * y + x * 0): the naive, unsimplified product rule.
        assert_eq!(g.to_expr_string(d), "cos(x * y) * (1 * y + x * 0)");
    }

    #[test]
    fn round_trips_by_value() {
        // Print a compiled expression, re-compile the printed form, and confirm
        // both evaluate identically at random points. Uses only non-negative
        // literal exponents so the formula lowers back through the front end.
        let sources = [
            "x * y + sin(x)",
            "(x + y) * z - x / y",
            "x^2 + 2*x + 1",
            "cos(x * y) + exp(x)",
            "-x + y^3",
        ];
        let mut rng = rand::rng();
        for src in sources {
            let (mut g, root) = compile(src).expect("compile source");
            let printed = g.to_expr_string(root);
            let (mut g2, root2) = compile(&printed)
                .unwrap_or_else(|e| panic!("re-compile {printed:?} failed: {e:?}"));

            for _ in 0..10 {
                let point: HashMap<String, f64> = ["x", "y", "z"]
                    .iter()
                    .map(|v| (v.to_string(), rng.random_range(0.5..3.0)))
                    .collect();
                g.forward(&point).expect("forward original");
                g2.forward(&point).expect("forward reprinted");
                let a = g.nodes[root].value;
                let b = g2.nodes[root2].value;
                assert!(
                    (a - b).abs() < 1e-9,
                    "round-trip mismatch for {src:?} -> {printed:?}: {a} vs {b}"
                );
            }
        }
    }
}
