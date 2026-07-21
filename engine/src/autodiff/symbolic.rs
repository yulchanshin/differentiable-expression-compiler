//! Symbolic differentiation as a graph-to-graph transform.
//!
//! [`Graph::diff`] recurses an expression applying the textbook rules (sum,
//! product, quotient, power, chain) and appends a new subgraph for the
//! derivative, returning its root. It reuses the original nodes as
//! subexpressions and never mutates them, so the arena only grows and the
//! derivative root is the last node pushed.
//!
//! It deliberately does not memoize: the product rule duplicates subtrees, so
//! the derivative graph swells, which is the point (TICKET-450). The Phase 4
//! optimizer (`const_fold`, `cse`, `dce`) claws some of it back, but only
//! partially: `const_fold` folds all-constant nodes, not identities, so `x * 1`
//! and `x + 0` survive absent a dedicated simplification pass.

use crate::graph::arena::Graph;
use crate::graph::node::OpType;

impl Graph {
    /// Build a new subgraph computing `d(node)/d(wrt)` and return its root index.
    ///
    /// Appends to the arena (reusing existing nodes as subexpressions) and leaves
    /// every existing node untouched. The returned index is the derivative's
    /// root; evaluate it with [`Graph::forward`] and read `nodes[root].value`.
    pub fn diff(&mut self, node: usize, wrt: &str) -> usize {
        // Copy the op and inputs out so the immutable borrow of the arena ends
        // before we start pushing nodes (the builder helpers take `&mut self`).
        let op = self.nodes[node].op.clone();
        let inputs = self.nodes[node].inputs.clone();

        match op {
            // d/dx c = 0.
            OpType::Const(_) => self.constant(0.0),
            // d/dx x = 1; d/dx y = 0.
            OpType::Var(name) => self.constant(if name == wrt { 1.0 } else { 0.0 }),

            // d(a + b) = da + db.
            OpType::Add => {
                let da = self.diff(inputs[0], wrt);
                let db = self.diff(inputs[1], wrt);
                self.add(da, db)
            }
            // d(a - b) = da - db.
            OpType::Sub => {
                let da = self.diff(inputs[0], wrt);
                let db = self.diff(inputs[1], wrt);
                self.sub(da, db)
            }
            // d(-a) = -da.
            OpType::Neg => {
                let da = self.diff(inputs[0], wrt);
                self.neg(da)
            }
            // Product rule: d(a*b) = da*b + a*db.
            OpType::Mul => {
                let (a, b) = (inputs[0], inputs[1]);
                let da = self.diff(a, wrt);
                let db = self.diff(b, wrt);
                let left = self.mul(da, b);
                let right = self.mul(a, db);
                self.add(left, right)
            }
            // Quotient rule: d(a/b) = (da*b - a*db) / b^2.
            OpType::Div => {
                let (a, b) = (inputs[0], inputs[1]);
                let da = self.diff(a, wrt);
                let db = self.diff(b, wrt);
                let num_left = self.mul(da, b);
                let num_right = self.mul(a, db);
                let num = self.sub(num_left, num_right);
                let den = self.pow(b, 2.0);
                self.div(num, den)
            }
            // Power + chain: d(a^k) = k * a^(k-1) * da.
            OpType::Pow(k) => {
                let a = inputs[0];
                let da = self.diff(a, wrt);
                let coeff = self.constant(k);
                let a_pow = self.pow(a, k - 1.0);
                let outer = self.mul(coeff, a_pow);
                self.mul(outer, da)
            }
            // Chain rule: d(sin a) = cos(a) * da.
            OpType::Sin => {
                let a = inputs[0];
                let da = self.diff(a, wrt);
                let ca = self.cos(a);
                self.mul(ca, da)
            }
            // Chain rule: d(cos a) = -sin(a) * da.
            OpType::Cos => {
                let a = inputs[0];
                let da = self.diff(a, wrt);
                let sa = self.sin(a);
                let neg_sa = self.neg(sa);
                self.mul(neg_sa, da)
            }
            // Chain rule: d(exp a) = exp(a) * da.
            OpType::Exp => {
                let a = inputs[0];
                let da = self.diff(a, wrt);
                let ea = self.exp(a);
                self.mul(ea, da)
            }
            // Chain rule: d(ln a) = (1/a) * da.
            OpType::Ln => {
                let a = inputs[0];
                let da = self.diff(a, wrt);
                let one = self.constant(1.0);
                let inv = self.div(one, a);
                self.mul(inv, da)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Differentiate the expression built by `build` w.r.t. `wrt`, evaluate the
    // derivative node at `point`, and return its value. Reads the derivative
    // root's value explicitly rather than relying on forward's last-node return.
    fn deriv_at(build: impl Fn(&mut Graph) -> usize, wrt: &str, point: &[(&str, f64)]) -> f64 {
        let mut g = Graph::new();
        let root = build(&mut g);
        let d = g.diff(root, wrt);
        let map: HashMap<String, f64> = point.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        g.forward(&map).expect("forward should succeed");
        g.nodes[d].value
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn const_derivative_is_zero() {
        approx(deriv_at(|g| g.constant(5.0), "x", &[("x", 2.0)]), 0.0);
    }

    #[test]
    fn var_derivative_is_one_or_zero() {
        approx(deriv_at(|g| g.var("x".into()), "x", &[("x", 2.0)]), 1.0);
        // A variable other than `wrt` differentiates to 0. Include x in the
        // point so forward has a value for it even though d/dy x = 0.
        approx(
            deriv_at(
                |g| {
                    g.var("x".into());
                    g.var("y".into())
                },
                "x",
                &[("x", 2.0), ("y", 3.0)],
            ),
            0.0,
        );
    }

    #[test]
    fn sum_and_difference() {
        // d/dx (x + y) = 1.
        approx(
            deriv_at(
                |g| {
                    let x = g.var("x".into());
                    let y = g.var("y".into());
                    g.add(x, y)
                },
                "x",
                &[("x", 1.0), ("y", 2.0)],
            ),
            1.0,
        );
        // d/dy (x - y) = -1.
        approx(
            deriv_at(
                |g| {
                    let x = g.var("x".into());
                    let y = g.var("y".into());
                    g.sub(x, y)
                },
                "y",
                &[("x", 1.0), ("y", 2.0)],
            ),
            -1.0,
        );
    }

    #[test]
    fn negation() {
        // d/dx (-x) = -1.
        approx(
            deriv_at(
                |g| {
                    let x = g.var("x".into());
                    g.neg(x)
                },
                "x",
                &[("x", 4.0)],
            ),
            -1.0,
        );
    }

    #[test]
    fn product_rule() {
        // d/dx (x*y) = y = 3.
        approx(
            deriv_at(
                |g| {
                    let x = g.var("x".into());
                    let y = g.var("y".into());
                    g.mul(x, y)
                },
                "x",
                &[("x", 2.0), ("y", 3.0)],
            ),
            3.0,
        );
    }

    #[test]
    fn quotient_rule() {
        // d/dx (x/y) = 1/y = 0.5.
        approx(
            deriv_at(
                |g| {
                    let x = g.var("x".into());
                    let y = g.var("y".into());
                    g.div(x, y)
                },
                "x",
                &[("x", 3.0), ("y", 2.0)],
            ),
            0.5,
        );
        // d/dy (x/y) = -x/y^2 = -3/4.
        approx(
            deriv_at(
                |g| {
                    let x = g.var("x".into());
                    let y = g.var("y".into());
                    g.div(x, y)
                },
                "y",
                &[("x", 3.0), ("y", 2.0)],
            ),
            -0.75,
        );
    }

    #[test]
    fn power_rule() {
        // d/dx x^3 = 3x^2 = 12 at x = 2.
        approx(
            deriv_at(
                |g| {
                    let x = g.var("x".into());
                    g.pow(x, 3.0)
                },
                "x",
                &[("x", 2.0)],
            ),
            12.0,
        );
    }

    #[test]
    fn unary_chain_rules() {
        let x = 0.5_f64;
        // d/dx sin(x) = cos(x).
        approx(
            deriv_at(
                |g| {
                    let v = g.var("x".into());
                    g.sin(v)
                },
                "x",
                &[("x", x)],
            ),
            x.cos(),
        );
        // d/dx cos(x) = -sin(x).
        approx(
            deriv_at(
                |g| {
                    let v = g.var("x".into());
                    g.cos(v)
                },
                "x",
                &[("x", x)],
            ),
            -x.sin(),
        );
        // d/dx exp(x) = exp(x).
        approx(
            deriv_at(
                |g| {
                    let v = g.var("x".into());
                    g.exp(v)
                },
                "x",
                &[("x", x)],
            ),
            x.exp(),
        );
        // d/dx ln(x) = 1/x, at x = 2.
        approx(
            deriv_at(
                |g| {
                    let v = g.var("x".into());
                    g.ln(v)
                },
                "x",
                &[("x", 2.0)],
            ),
            0.5,
        );
    }

    #[test]
    fn nested_chain_and_product() {
        // f = sin(x*y);  d/dx f = cos(x*y) * y  (chain over the product).
        let (xv, yv) = (1.5_f64, 2.0_f64);
        approx(
            deriv_at(
                |g| {
                    let x = g.var("x".into());
                    let y = g.var("y".into());
                    let xy = g.mul(x, y);
                    g.sin(xy)
                },
                "x",
                &[("x", xv), ("y", yv)],
            ),
            (xv * yv).cos() * yv,
        );
    }

    #[test]
    fn diff_appends_and_leaves_original_intact() {
        // diff grows the arena and does not disturb the original expression:
        // f = x*y is still evaluable at its own root after differentiating.
        let mut g = Graph::new();
        let x = g.var("x".into());
        let y = g.var("y".into());
        let f = g.mul(x, y);
        let len_before = g.nodes.len();

        let d = g.diff(f, "x");
        assert!(g.nodes.len() > len_before, "diff should append nodes");

        let map = HashMap::from([("x".to_string(), 2.0), ("y".to_string(), 3.0)]);
        g.forward(&map).expect("forward should succeed");
        approx(g.nodes[f].value, 6.0); // original f = x*y still intact
        approx(g.nodes[d].value, 3.0); // derivative df/dx = y
    }
}
