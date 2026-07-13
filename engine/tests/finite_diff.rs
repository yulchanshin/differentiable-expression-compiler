//! Finite-difference oracle: the AD correctness harness (TICKET-104).
//!
//! Integration test: compiles as its own crate and exercises `engine`
//! from the outside via `use engine::...`. The real oracle lands in
//! TICKET-104 once the autodiff core exists; this file just stakes out
//! the location.

use engine::graph::arena::Graph;
use rand::RngExt;
use std::collections::HashMap;

/// Estimate the gradient of `f` at `point` by central differences:
/// `∂f/∂xᵢ ≈ (f(x+h) − f(x−h)) / 2h`, nudging one variable at a time.
///
/// This is the numerical oracle: a black box that only ever calls `f`, so
/// it shares no code with the AD engine and can independently validate it.
fn numerical_gradient(
    mut f: impl FnMut(&HashMap<String, f64>) -> f64,
    point: &HashMap<String, f64>,
    h: f64,
) -> HashMap<String, f64> {
    let mut grad: HashMap<String, f64> = HashMap::new();
    let mut modified: HashMap<String, f64> = point.clone();
    for key in point.keys() {
        let base: f64 = point[key];
        modified.insert(key.clone(), base + h);
        let f_plus: f64 = f(&modified);
        modified.insert(key.clone(), base - h);
        let f_minus: f64 = f(&modified);
        modified.insert(key.clone(), base); // restore
        grad.insert(key.clone(), (f_plus - f_minus) / (2.0 * h));
    }
    grad
}

fn check(build: impl Fn(&mut Graph) -> usize, point: &HashMap<String, f64>, tolerance: f64) {
    //Automatic Differentiation
    let mut g = Graph::new();
    build(&mut g);
    g.forward(point).expect("forward should succeed");
    let auto_diff: HashMap<String, f64> = g
        .backward(g.nodes.len() - 1)
        .expect("backward should succeed");

    //Numerical Differentiation
    let f = |inputs: &HashMap<String, f64>| {
        let mut gg = Graph::new();
        build(&mut gg);
        gg.forward(inputs).expect("forward should succeed")
    };

    let num_grad: HashMap<String, f64> = numerical_gradient(f, point, H);

    for key in point.keys() {
        assert!(
            (auto_diff[key] - num_grad[key]).abs() < tolerance,
            "grad mismatch on {key}: auto_diff={}, num_grad={}",
            auto_diff[key],
            num_grad[key],
        );
    }
}

const TOLERANCE: f64 = 1e-6; // AD-vs-numerical agreement threshold
const H: f64 = 1e-5; // central-difference step size
const N_POINTS: usize = 10; // random points probed per expression

// Probe `build` at N_POINTS random points, each variable in `vars` drawn
// uniformly from [low, high), asserting AD matches the numerical oracle.
// `build` needs `+ Copy` because `check` takes it by value and we call it
// once per point; non-capturing closures are Copy, so this is satisfied.
fn check_random(build: impl Fn(&mut Graph) -> usize + Copy, vars: &[&str], low: f64, high: f64) {
    let mut rng = rand::rng();
    for _ in 0..N_POINTS {
        let point: HashMap<String, f64> = vars
            .iter()
            .map(|v| (v.to_string(), rng.random_range(low..high)))
            .collect();
        check(build, &point, TOLERANCE);
    }
}

// x + 1
#[test]
fn add() {
    let build = |g: &mut Graph| {
        let c: usize = g.constant(1.0);
        let x: usize = g.var("x".into());
        g.add(x, c)
    };
    check_random(build, &["x"], 0.0, 100.0);
}
// (xy) ^ 2
#[test]
fn pow() {
    let build = |g: &mut Graph| {
        let x: usize = g.var("x".into());
        let y: usize = g.var("y".into());
        let xy: usize = g.mul(x, y);
        g.pow(xy, 2.0)
    };
    check_random(build, &["x", "y"], 0.1, 1.0);
}
// ln(cos(xy))
#[test]
fn ln_cos_xy() {
    let build = |g: &mut Graph| {
        let x: usize = g.var("x".into());
        let y: usize = g.var("y".into());
        let xy: usize = g.mul(x, y);
        let cos_xy: usize = g.cos(xy);
        g.ln(cos_xy)
    };
    check_random(build, &["x", "y"], 0.1, 1.0);
}
// cos(sin(cos(exp(x))))
#[test]
fn wrapped_trig() {
    let build = |g: &mut Graph| {
        let x: usize = g.var("x".into());
        let exp: usize = g.exp(x);
        let cos_exp_x: usize = g.cos(exp);
        let sin_cos_exp_x: usize = g.sin(cos_exp_x);
        g.cos(sin_cos_exp_x)
    };
    check_random(build, &["x"], 0.1, 1.0);
}
// sin(x) / (x^2 + y^2)
#[test]
fn fraction() {
    let build = |g: &mut Graph| {
        let x: usize = g.var("x".into());
        let y: usize = g.var("y".into());
        let x_sqr: usize = g.pow(x, 2.0);
        let y_sqr: usize = g.pow(y, 2.0);
        let x_sqr_plus_y_sqr: usize = g.add(x_sqr, y_sqr);
        let sin_x: usize = g.sin(x);
        g.div(sin_x, x_sqr_plus_y_sqr)
    };
    check_random(build, &["x", "y"], 0.1, 1.0);
}
// -exp(x^2y^3)
#[test]
fn neg_exp() {
    let build = |g: &mut Graph| {
        let x: usize = g.var("x".into());
        let y: usize = g.var("y".into());
        let x_sqr: usize = g.pow(x, 2.0);
        let y_cube: usize = g.pow(y, 3.0);
        let x_sqr_times_y_cube: usize = g.mul(x_sqr, y_cube);
        let exp: usize = g.exp(x_sqr_times_y_cube);
        g.neg(exp)
    };
    check_random(build, &["x", "y"], 0.1, 1.0);
}
// xyz + sin(xy) + ln(y^2)
#[test]
fn triple_var() {
    let build = |g: &mut Graph| {
        let x: usize = g.var("x".into());
        let y: usize = g.var("y".into());
        let z: usize = g.var("z".into());
        let y_sqr = g.pow(y, 2.0);
        let xy: usize = g.mul(x, y);
        let xyz: usize = g.mul(xy, z);
        let sin_xy: usize = g.sin(xy);
        let ln_y_sqr: usize = g.ln(y_sqr);
        let xyz_plus_sin_xy: usize = g.add(xyz, sin_xy);
        g.add(xyz_plus_sin_xy, ln_y_sqr)
    };
    check_random(build, &["x", "y", "z"], 0.1, 1.0);
}
// ln(x^2 * y) + sin(exp(xy)) + x^2 + y^2 + z^2
#[test]
fn crazy() {
    let build = |g: &mut Graph| {
        let x: usize = g.var("x".into());
        let y: usize = g.var("y".into());
        let z: usize = g.var("z".into());

        let x_sqr: usize = g.pow(x, 2.0);
        let y_sqr: usize = g.pow(y, 2.0);
        let z_sqr: usize = g.pow(z, 2.0);

        let x_sqr_y: usize = g.mul(x_sqr, y);
        let xy: usize = g.mul(x, y);
        let exp_xy: usize = g.exp(xy);

        let ln_x_sqr_y: usize = g.ln(x_sqr_y);
        let sin_exp_xy: usize = g.sin(exp_xy);
        let x_sqr_plus_y_sqr: usize = g.add(x_sqr, y_sqr);
        let x_sqr_plus_y_sqr_plus_z_sqr: usize = g.add(x_sqr_plus_y_sqr, z_sqr);

        let ln_plus_sin: usize = g.add(ln_x_sqr_y, sin_exp_xy);

        g.add(ln_plus_sin, x_sqr_plus_y_sqr_plus_z_sqr)
    };
    check_random(build, &["x", "y", "z"], 0.1, 1.0);
}
// exp(x) - ln(y)
#[test]
fn sub() {
    let build = |g: &mut Graph| {
        let x: usize = g.var("x".into());
        let y: usize = g.var("y".into());
        let exp_x: usize = g.exp(x);
        let ln_y: usize = g.ln(y);
        g.sub(exp_x, ln_y)
    };
    check_random(build, &["x", "y"], 0.1, 1.0);
}

// Proves the oracle BITES: if a computed gradient disagrees with the
// finite-difference estimate, the comparison must panic. Here f(x) = x^2 has
// true gradient 2x (= 6 at x=3), but we assert a deliberately wrong value of
// 0.0 against the numerical estimate. The oracle should reject it. This is
// the permanent stand-in for "temporarily break a derivative and watch a
// test fail" — it guarantees the assertion actually fires on mismatch.
#[test]
#[should_panic(expected = "grad mismatch")]
fn oracle_catches_wrong_gradient() {
    let f = |inputs: &HashMap<String, f64>| {
        let x: f64 = inputs["x"];
        x * x
    };
    let point: HashMap<String, f64> = HashMap::from([("x".into(), 3.0)]);
    let num: HashMap<String, f64> = numerical_gradient(f, &point, H);

    let wrong: f64 = 0.0; // a broken "derivative" of x^2
    assert!(
        (wrong - num["x"]).abs() < TOLERANCE,
        "grad mismatch on x: wrong={}, num={}",
        wrong,
        num["x"],
    );
}
