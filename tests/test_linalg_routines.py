"""
Direct unit tests for the numerical routines in QuantumSCC.utils.linalg
(Tier A).

test_linalg.py already covers GaussJordan and the symplectic-form transforms
at a high level; this module exercises the individual routines that were only
reached indirectly through the pipeline: integer_null_space, pseudo_inv,
proportional_rows, Gauge_variable_symplification, remove_zero_rows, and the
edge branches of omega_symplectic_transformation / symplectic_transformation.
"""

from math import gcd

import numpy as np
import pytest
from scipy.linalg import null_space

from QuantumSCC.utils.linalg import (
    Gauge_variable_symplification,
    GaussJordan,
    integer_null_space,
    omega_symplectic_transformation,
    proportional_rows,
    pseudo_inv,
    remove_zero_rows,
    reverseGaussJordan,
    symplectic_transformation,
)


def _J2(nf):
    return np.block([[np.zeros((nf, nf)), np.eye(nf)],
                     [-np.eye(nf), np.zeros((nf, nf))]])


# ── integer_null_space ──────────────────────────────────────────────────────

class TestIntegerNullSpace:
    def test_kernel_is_exact(self):
        M = np.array([[1, -1, 0], [0, 1, -1]], float)
        K = integer_null_space(M)
        np.testing.assert_allclose(M @ K, 0, atol=1e-12)

    def test_entries_are_integers(self):
        M = np.array([[2, -2, 0]], float)
        K = integer_null_space(M)
        np.testing.assert_allclose(K, np.round(K), atol=1e-12)

    def test_columns_are_coprime(self):
        M = np.array([[1, -1, 0]], float)
        K = integer_null_space(M)
        for j in range(K.shape[1]):
            entries = [abs(int(round(x))) for x in K[:, j] if abs(x) > 1e-9]
            g = 0
            for e in entries:
                g = gcd(g, e)
            assert g == 1

    def test_kernel_dimension_matches_rank_nullity(self):
        M = np.array([[1, -1, 0], [0, 1, -1]], float)
        K = integer_null_space(M)
        assert K.shape[1] == M.shape[1] - np.linalg.matrix_rank(M)

    def test_trivial_kernel_is_empty(self):
        K = integer_null_space(np.eye(3))
        assert K.shape == (3, 0)

    def test_empty_matrix_returns_identity(self):
        K = integer_null_space(np.zeros((0, 3)))
        np.testing.assert_allclose(K, np.eye(3))


# ── pseudo_inv ──────────────────────────────────────────────────────────────

class TestPseudoInv:
    def test_moore_penrose_conditions(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Ap = pseudo_inv(A)
        np.testing.assert_allclose(A @ Ap @ A, A, atol=1e-9)
        np.testing.assert_allclose(Ap @ A @ Ap, Ap, atol=1e-9)

    def test_matches_numpy_pinv_well_conditioned(self):
        A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
        np.testing.assert_allclose(pseudo_inv(A), np.linalg.pinv(A), atol=1e-6)

    def test_rank_deficient(self):
        A = np.array([[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(pseudo_inv(A), np.array([[1.0, 0.0], [0.0, 0.0]]),
                                   atol=1e-12)

    def test_tolerance_drops_tiny_singular_value(self):
        U, _ = np.linalg.qr(np.random.RandomState(0).randn(3, 3))
        A = U @ np.diag([1.0, 0.5, 1e-13]) @ U.T
        Ap = pseudo_inv(A, tol=1e-10)
        assert np.linalg.matrix_rank(Ap) == 2


# ── proportional_rows ───────────────────────────────────────────────────────

class TestProportionalRows:
    def test_identifies_proportional_group(self):
        M = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 1.0]])
        assert proportional_rows(M) == [[0, 1]]

    def test_no_proportional_rows(self):
        assert proportional_rows(np.eye(3)) == []

    def test_multiple_groups(self):
        M = np.array([[1.0, 0.0], [2.0, 0.0],   # group A (rows 0,1)
                      [0.0, 1.0], [0.0, 3.0]])   # group B (rows 2,3)
        groups = proportional_rows(M)
        assert [0, 1] in groups
        assert [2, 3] in groups

    def test_antiproportional_rows_grouped(self):
        # opposite-sign multiples are still proportional (constant ratio).
        M = np.array([[1.0, 2.0], [-2.0, -4.0]])
        assert proportional_rows(M) == [[0, 1]]

    def test_nonproportional_all_nonzero(self):
        # all-nonzero rows with a differing column ratio -> not proportional
        # (exercises the ratio-mismatch branch).
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert proportional_rows(M) == []


# ── Gauge_variable_symplification ───────────────────────────────────────────

class TestGaugeSimplification:
    def test_row_zeroed_except_pivot(self):
        M = np.array([[2.0, 4.0], [1.0, 3.0]])
        out = Gauge_variable_symplification(M, row_index=0, column_index=0)
        # pivot normalised to 1, the rest of the row eliminated to 0.
        assert abs(out[0, 0] - 1.0) < 1e-12
        assert abs(out[0, 1]) < 1e-12

    def test_zero_pivot_raises(self):
        M = np.array([[0.0, 4.0], [1.0, 3.0]])
        with pytest.raises(ValueError):
            Gauge_variable_symplification(M, row_index=0, column_index=0)

    def test_does_not_mutate_input_dtype_int(self):
        # integer input is cast to float internally (astype), original unchanged
        M = np.array([[2, 4], [1, 3]])
        Gauge_variable_symplification(M, 0, 0)
        assert M.dtype == np.dtype(int)


# ── remove_zero_rows ────────────────────────────────────────────────────────

class TestRemoveZeroRows:
    def test_removes_zero_rows(self):
        M = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]])
        np.testing.assert_allclose(remove_zero_rows(M), [[1.0, 2.0]])

    def test_keeps_all_when_none_zero(self):
        M = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(remove_zero_rows(M), M)

    def test_tolerance(self):
        M = np.array([[1e-20, 1e-20], [1.0, 0.0]])
        # first row is below the default tol -> dropped
        np.testing.assert_allclose(remove_zero_rows(M), [[1.0, 0.0]])


# ── GaussJordan / reverseGaussJordan ────────────────────────────────────────

class TestGaussJordanExtra:
    def test_returns_matrix_and_order(self):
        M = np.array([[0.0, 1.0], [1.0, 0.0]])
        out, order = GaussJordan(M)
        assert out.shape == M.shape
        assert sorted(order.tolist()) == [0, 1]

    def test_does_not_mutate_input(self):
        M = np.array([[2.0, 1.0], [1.0, 3.0]])
        M_copy = M.copy()
        GaussJordan(M)
        np.testing.assert_allclose(M, M_copy)

    def test_reverse_diagonalises(self):
        # reverseGaussJordan normalises the diagonal to 1 and clears above it.
        M = np.array([[-1.0, 0.0, 0.0, -1.0, 1.0],
                      [0.0, -1.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, -1.0, 0.0, 1.0]])
        out = reverseGaussJordan(M)
        np.testing.assert_allclose(np.diag(out[:, :3]), [1.0, 1.0, 1.0])


# ── omega_symplectic_transformation ─────────────────────────────────────────

class TestOmegaSymplectic:
    def test_canonical_invariant(self):
        Om = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                       [-1, 0, 0, 0], [0, -1, 0, 0]], float)
        J, V, nCF, nCC = omega_symplectic_transformation(
            Om, no_compact_flux_variables=0, no_flux_variables=2)
        np.testing.assert_allclose(J, V.T @ Om @ V, atol=1e-10)

    def test_returns_four_values(self):
        Om = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                       [-1, 0, 0, 0], [0, -1, 0, 0]], float)
        result = omega_symplectic_transformation(
            Om, no_compact_flux_variables=0, no_flux_variables=2)
        assert len(result) == 4
        assert isinstance(result[2], int)
        assert isinstance(result[3], int)

    def test_non_antisymmetric_input_raises(self):
        with pytest.raises(AssertionError):
            omega_symplectic_transformation(
                np.eye(4), no_compact_flux_variables=0, no_flux_variables=2)


# ── symplectic_transformation ───────────────────────────────────────────────

class TestSymplecticTransformation:
    def test_oscillator_only_is_symplectic_and_real(self):
        H = np.array([[0.6, 0.15, 0.0, 0.0],
                      [0.15, 0.6, 0.0, 0.0],
                      [0.0, 0.0, 0.6, -0.15],
                      [0.0, 0.0, -0.15, 0.6]])
        J = _J2(2)
        _, T = symplectic_transformation(J @ H, no_flux_variables=2)
        np.testing.assert_allclose(T.T @ J @ T, J, atol=1e-9)
        np.testing.assert_allclose(T.imag, 0.0, atol=1e-9)

    def test_zero_mode_branch(self):
        # One oscillator mode + one zero-frequency (free) mode exercises the
        # n_zero > 0 branch (flux/charge null-space pairing).
        H = np.diag([1.0, 0.0, 1.0, 0.0])
        J = _J2(2)
        _, T = symplectic_transformation(J @ H, no_flux_variables=2)
        np.testing.assert_allclose(T.T @ J @ T, J, atol=1e-9)
        np.testing.assert_allclose(T.imag, 0.0, atol=1e-9)

    def test_odd_dimension_raises(self):
        with pytest.raises(AssertionError):
            symplectic_transformation(np.zeros((3, 3)), no_flux_variables=1)


# ── differential tests against numpy / scipy reference implementations ───────

class TestAgainstNumpyScipy:
    """Cross-validate the custom routines against the consolidated numpy/scipy
    implementations, not just against internal mathematical invariants."""

    @pytest.mark.parametrize("M", [
        np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]),
        np.array([[1.0, 1.0, 1.0, 1.0]]),
        np.array([[2.0, -2.0, 0.0, 0.0], [0.0, 0.0, 3.0, -3.0]]),
        np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
    ])
    def test_integer_null_space_matches_scipy_null_space(self, M):
        """integer_null_space spans the same subspace as scipy.linalg.null_space
        (same dimension; every integer-kernel column lies in scipy's null space)."""
        K = integer_null_space(M)
        N = null_space(M)
        assert K.shape[1] == N.shape[1]
        if N.shape[1] > 0:
            # N is orthonormal -> N N^T is the projector onto the null space;
            # K's columns are unchanged by it iff they live in that subspace.
            np.testing.assert_allclose(N @ (N.T @ K), K, atol=1e-9)

    @pytest.mark.parametrize("A", [
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),               # rank deficient
        np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]),
    ])
    def test_pseudo_inv_matches_numpy_pinv(self, A):
        np.testing.assert_allclose(pseudo_inv(A), np.linalg.pinv(A), atol=1e-6)

    @pytest.mark.parametrize("M", [
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]]),
        np.random.RandomState(1).randn(4, 6),
    ])
    def test_gaussjordan_rank_matches_numpy(self, M):
        """The number of non-zero rows after Gauss-Jordan equals numpy's rank."""
        reduced, _ = GaussJordan(M)
        assert remove_zero_rows(reduced).shape[0] == np.linalg.matrix_rank(M)

    def test_symplectic_frequencies_match_numpy_eig(self):
        """The mode frequencies produced by symplectic_transformation equal the
        positive imaginary parts of the eigenvalues of J·H (numpy.linalg.eig)."""
        H = np.diag([0.6, 0.9, 0.6, 0.9])
        J = _J2(2)
        M_out, _ = symplectic_transformation(J @ H, no_flux_variables=2)
        # M_out has the block form [[0, Ω], [-Ω, 0]] with Ω = diag(ω_i).
        freqs_routine = np.sort(np.abs(np.diag(M_out[:2, 2:])))
        eig = np.linalg.eigvals(J @ H)
        freqs_ref = np.sort(np.abs(eig.imag[eig.imag > 1e-9]))
        np.testing.assert_allclose(freqs_routine, freqs_ref, atol=1e-9)
