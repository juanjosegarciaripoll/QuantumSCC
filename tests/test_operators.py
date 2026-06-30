"""Unit tests for the per-mode sparse operators."""

import math

import numpy as np
import scipy.sparse as sp

from QuantumSCC.diag import charge_mode, dual_charge_mode, oscillator_mode
from QuantumSCC.diag.operators import displacement


def test_oscillator_canonical_commutator():
    # [φ, n] = i on the truncated space, away from the top Fock level.
    m = oscillator_mode(20)
    comm = (m.flux @ m.charge - m.charge @ m.flux).toarray()
    np.testing.assert_allclose(np.diag(comm)[:-1], 1j, atol=1e-12)


def test_oscillator_number_operator():
    # ½(φ² + n²) = a†a + ½, so the diagonal is 0.5, 1.5, 2.5, ... away from the
    # truncated top level, where φ² loses the |dim> contribution.
    m = oscillator_mode(6)
    energy = 0.5 * (m.flux @ m.flux + m.charge @ m.charge).toarray()
    np.testing.assert_allclose(np.diag(energy).real[:-1], np.arange(5) + 0.5, atol=1e-12)


def test_charge_mode_shift_is_unitary_and_raises():
    m = charge_mode(3)  # n in [-3, 3]
    shift = m.exp_flux(1.0)  # exp(iφ): n -> n+1
    # |n=0> is index 3; shifting once must populate index 4 (n=1).
    vec = np.zeros(m.dim)
    vec[3] = 1.0
    out = shift @ vec
    assert np.argmax(np.abs(out)) == 4


def test_charge_mode_exp_charge_diagonal():
    m = charge_mode(2)
    op = m.exp_charge(0.7)
    dense = op.toarray()
    assert np.count_nonzero(dense - np.diag(np.diag(dense))) == 0
    np.testing.assert_allclose(np.diag(dense), np.exp(1j * 0.7 * np.arange(-2, 3)))


def test_displacement_zero_is_identity():
    D = displacement(0.0, 8).toarray()
    np.testing.assert_allclose(D, np.eye(8), atol=1e-14)


def test_displacement_vacuum_is_exact_coherent_state():
    # The vacuum column is exact for every retained level: exp(−α*a)|0> = |0>
    # and exp(α a†)|0> terminates, so <n|D(α)|0> = e^{−|α|²/2} αⁿ/√(n!) with no
    # truncation error. Tolerance is fixed by series round-off, not by expm.
    dim, alpha = 24, 0.7 + 0.3j
    amplitudes = displacement(alpha, dim).toarray()[:, 0]
    n = np.arange(dim)
    sqrt_factorial = np.array([math.sqrt(math.factorial(k)) for k in n])
    exact = np.exp(-0.5 * abs(alpha) ** 2) * alpha**n / sqrt_factorial
    np.testing.assert_allclose(amplitudes, exact, atol=1e-12)


def test_displacement_unitary_in_bulk():
    # D is exactly unitary on the infinite space; on the truncated space the
    # defect is the probability displaced past level `dim`. For |α|=1 and a bulk
    # well below the cutoff that leakage is far under 1e-6.
    dim, bulk = 30, 12
    D = displacement(1.0j, dim).toarray()
    gram = (D.conj().T @ D)[:bulk, :bulk]
    np.testing.assert_allclose(gram, np.eye(bulk), atol=1e-6)


def test_displacement_inverse_is_negative_argument():
    # D(α)† = D(−α) exactly; check on the (exact) vacuum column.
    dim, alpha = 24, 0.9j
    forward = displacement(alpha, dim).toarray()
    backward = displacement(-alpha, dim).toarray()
    np.testing.assert_allclose(forward.conj().T[:, 0], backward[:, 0], atol=1e-12)


def test_dual_is_mirror_of_charge_mode():
    # In the QPS sector flux <-> charge roles swap.
    c = charge_mode(2)
    d = dual_charge_mode(2)
    assert c.flux is None and c.charge is not None
    assert d.charge is None and d.flux is not None
    assert (sp.csc_matrix(c.exp_charge(0.4)) - d.exp_flux(0.4)).nnz == 0
