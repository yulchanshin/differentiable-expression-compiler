//! Dense linear solver: `A x = b` via LU decomposition with partial pivoting.
//!
//! The inner solve of Newton's method (TICKET-501): `J·Δx = −f(x)` every step.
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

/// Factor `a` into `(L, U, piv)` with partial pivoting so that `P A = L U`.
///
/// `L` is unit-lower-triangular, `U` is upper-triangular, and `piv` is the row
/// permutation (original row `piv[i]` ends up at row `i`). Returns
/// [`EngineError::SingularMatrix`] if a pivot is ~0 even after the swap.
pub fn lu_decompose(
    a: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>), EngineError> {
    // Plan (Doolittle with partial pivoting):
    //   1. n = a.len(); start piv = [0, 1, …, n-1]; copy `a` into a working
    //      matrix `m` you can overwrite in place.
    //   2. For each column k in 0..n:
    //        a. Find the row p in k..n whose |m[p][k]| is largest (the pivot).
    //        b. If |m[p][k]| < PIVOT_EPSILON → Err(SingularMatrix).
    //        c. Swap rows k and p in `m`, and swap piv[k]/piv[p] to record it.
    //        d. For each row i in (k+1)..n: factor = m[i][k] / m[k][k];
    //           store that multiplier at m[i][k], then eliminate across the row
    //           m[i][j] -= factor * m[k][j] for j in (k+1)..n.
    //   3. Split `m` into L (unit diagonal, multipliers below it) and U (the
    //      diagonal and above). Return (l, u, piv).
    todo!("Doolittle elimination with partial pivoting")
}

/// Solve `A x = b` given the factors from [`lu_decompose`].
///
/// Applies the row permutation to `b`, forward-substitutes `L y = P b`, then
/// back-substitutes `U x = y`. Infallible: singularity was already caught during
/// decomposition, so every `U[i][i]` here is nonzero.
pub fn lu_solve(l: &[Vec<f64>], u: &[Vec<f64>], piv: &[usize], b: &[f64]) -> Vec<f64> {
    // Plan:
    //   1. Permute: pb[i] = b[piv[i]].
    //   2. Forward solve L y = pb (L unit-lower): for i in 0..n,
    //        y[i] = pb[i] - Σ_{j<i} l[i][j] * y[j].
    //   3. Back solve U x = y: for i in (0..n).rev(),
    //        x[i] = (y[i] - Σ_{j>i} u[i][j] * x[j]) / u[i][i].
    todo!("forward + back substitution")
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
