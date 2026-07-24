//! Dense linear solver: `A x = b` via LU decomposition with partial pivoting.
//!
//! The inner solve of IK's damped-least-squares step (TICKET-502): each
//! iteration solves `(JJᵀ + λ²I)·y = e` for the tip-space error `e`.
//! Factor `A` once (O(n³)) into unit-lower `L` and upper `U`, then each solve is
//! two O(n²) triangular passes — forward `L y = b`, back `U x = y` — reusable
//! across right-hand sides. Partial pivoting swaps the largest-magnitude row
//! into each pivot before eliminating (else a zero/tiny pivot gives a wrong
//! answer); `piv` records it, giving `P A = L U` where `P` sends original row
//! `piv[i]` to row `i`, so `b` is permuted `(P b)[i] = b[piv[i]]`. A pivot still
//! ~0 after the swap means `A` is singular → [`EngineError::SingularMatrix`],
//! not an `inf`/`NaN`.
//!
//! Matrices are row-major `Vec<Vec<f64>>` (matching `Graph::jacobian`). `A` is
//! assumed square; a non-square `A` is a programmer bug, not a runtime error.

use crate::error::EngineError;

/// Pivot magnitude below which a pivot column is treated as dependent (singular).
const PIVOT_EPSILON: f64 = 1e-12;

/// The factors from [`lu_decompose`]: unit-lower `L`, upper `U`, and the row
/// permutation `piv`.
pub type LuFactors = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>);

/// Factor `a` into `(L, U, piv)` with partial pivoting so that `P A = L U`.
///
/// `L` is unit-lower-triangular, `U` is upper-triangular, and `piv` is the row
/// permutation (original row `piv[i]` ends up at row `i`). Returns
/// [`EngineError::SingularMatrix`] if a pivot is ~0 even after the swap.
pub fn lu_decompose(a: &[Vec<f64>]) -> Result<LuFactors, EngineError> {
    let n = a.len();
    // Work in place on a copy; `m` ends up holding L's multipliers below the
    // diagonal and U on/above it (the packed Doolittle form).
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Partial pivot: the largest-magnitude entry in column k over rows k..n.
        let p = (k..n)
            .max_by(|&i, &j| m[i][k].abs().total_cmp(&m[j][k].abs()))
            .unwrap();
        if m[p][k].abs() < PIVOT_EPSILON {
            return Err(EngineError::SingularMatrix);
        }
        m.swap(k, p);
        piv.swap(k, p);

        // Eliminate below the pivot, caching each multiplier where the zero it
        // creates would go.
        for i in (k + 1)..n {
            let factor = m[i][k] / m[k][k];
            m[i][k] = factor;
            for j in (k + 1)..n {
                m[i][j] -= factor * m[k][j];
            }
        }
    }

    // Unpack: L is unit-lower (multipliers below the diagonal), U is the
    // diagonal and above.
    let mut l = vec![vec![0.0; n]; n];
    let mut u = vec![vec![0.0; n]; n];
    for i in 0..n {
        l[i][i] = 1.0;
        l[i][..i].copy_from_slice(&m[i][..i]);
        u[i][i..].copy_from_slice(&m[i][i..]);
    }
    Ok((l, u, piv))
}

/// Solve `A x = b` given the factors from [`lu_decompose`].
///
/// Applies the row permutation to `b`, forward-substitutes `L y = P b`, then
/// back-substitutes `U x = y`. Infallible: singularity was already caught during
/// decomposition, so every `U[i][i]` here is nonzero.
pub fn lu_solve(l: &[Vec<f64>], u: &[Vec<f64>], piv: &[usize], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    // Apply the same row permutation the factoring used.
    let pb: Vec<f64> = piv.iter().map(|&i| b[i]).collect();

    // Forward solve L y = P b (L unit-lower, so no divide).
    let mut y = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|j| l[i][j] * y[j]).sum();
        y[i] = pb[i] - s;
    }

    // Back solve U x = y.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = ((i + 1)..n).map(|j| u[i][j] * x[j]).sum();
        x[i] = (y[i] - s) / u[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    // Dense matrix-vector product, so tests can check `A x ≈ b` directly.
    fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|row| row.iter().zip(x).map(|(aij, xj)| aij * xj).sum())
            .collect()
    }

    // Solve end to end: decompose then solve. Panics if decomposition errors.
    fn solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let (l, u, piv) = lu_decompose(a).expect("matrix should be nonsingular");
        lu_solve(&l, &u, &piv, b)
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9, "expected {want:?}, got {got:?}");
        }
    }

    // A random n×n matrix made strictly diagonally dominant, which guarantees it
    // is nonsingular and well-conditioned (so the 1e-9 target is achievable).
    fn diagonally_dominant(n: usize, rng: &mut impl RngExt) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            let mut off = 0.0;
            for j in 0..n {
                if i != j {
                    let v: f64 = rng.random_range(-1.0..1.0);
                    a[i][j] = v;
                    off += v.abs();
                }
            }
            // Diagonal strictly exceeds the row's off-diagonal sum.
            a[i][i] = off + rng.random_range(1.0..2.0);
        }
        a
    }

    #[test]
    fn solves_hand_checked_2x2() {
        // 2x + y = 5 ; x + 3y = 10  →  x = 1, y = 3.
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        assert_close(&solve(&a, &b), &[1.0, 3.0]);
    }

    #[test]
    fn identity_returns_b() {
        let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let b = vec![7.0, -2.0, 0.5];
        assert_close(&solve(&a, &b), &b);
    }

    #[test]
    fn requires_pivoting() {
        // A zero in the natural first pivot: only partial pivoting (swapping the
        // rows) makes this solvable.
        let a = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let b = vec![2.0, 3.0]; // y = 2, x + y = 3 → x = 1
        assert_close(&solve(&a, &b), &[1.0, 2.0]);
    }

    #[test]
    fn detects_singular() {
        // Row 2 = 2 * row 1: rank-deficient, no unique solution.
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(matches!(lu_decompose(&a), Err(EngineError::SingularMatrix)));
    }

    #[test]
    fn solves_random_well_conditioned_systems() {
        // Acceptance criterion: random well-conditioned A x = b to 1e-9.
        // Pick a known x, form b = A x, recover x, compare.
        let mut rng = rand::rng();
        for n in [1usize, 2, 3, 5, 8, 16] {
            let a = diagonally_dominant(n, &mut rng);
            let x_true: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
            let b = matvec(&a, &x_true);
            assert_close(&solve(&a, &b), &x_true);
        }
    }
}
