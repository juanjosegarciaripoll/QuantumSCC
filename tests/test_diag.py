"""
Tests for QuantumSCC.diag.

Validates the diagonaliser (building on QuantumSCC.CircuitModel) and the
reconstructed spectra against known physics.
"""

import numpy as np

from QuantumSCC import Capacitor, Circuit, Inductor, Junction, PhaseSlip
from QuantumSCC.diag import build_hamiltonian, eigenenergies, eigenstates


def transmon():
    return Circuit([(0, 1, Junction(10, "GHz")), (0, 1, Capacitor(0.2, "GHz"))])


def lc():
    return Circuit([(0, 1, Inductor(2, "GHz")), (0, 1, Capacitor(0.5, "GHz"))])


def dual_transmon():
    return Circuit([(0, 1, PhaseSlip(5, "GHz")), (0, 1, Inductor(1, "GHz"))])


def fluxonium():
    # Junction shunted by an inductor: a single *extended* mode whose cosine
    # exercises the oscillator displacement path (cos φ on a Fock space).
    return Circuit(
        [
            (0, 1, Junction(5, "GHz")),
            (0, 1, Inductor(1, "GHz")),
            (0, 1, Capacitor(1, "GHz")),
        ]
    )


def test_accepts_circuit_and_model():
    c = transmon()
    h1 = build_hamiltonian(c)
    h2 = build_hamiltonian(c.model())
    assert (h1 - h2).nnz == 0 or abs((h1 - h2).toarray()).max() < 1e-12


def test_lc_spectrum_equally_spaced():
    # harmonic oscillator: levels spaced by ω = sqrt(4 E_C E_L) = 2 GHz
    e = eigenenergies(lc(), k=4)
    np.testing.assert_allclose(np.diff(e), [2.0, 2.0, 2.0], atol=1e-6)


def test_transmon_transition_frequency():
    # plasma frequency sqrt(8 E_C E_J) = 2 GHz minus anharmonicity
    e = eigenenergies(transmon(), k=4, relative=True)
    assert abs(e[1] - 1.9487) < 1e-3
    assert e[0] == 0.0  # relative to ground state


def test_dual_transmon_has_finite_gap():
    e = eigenenergies(dual_transmon(), k=4, relative=True)
    assert e[1] > 0.0


def test_hermitian():
    H = build_hamiltonian(transmon())
    assert abs((H - H.conj().T).toarray()).max() < 1e-10


def test_eigenstates_consistent_with_eigenenergies():
    vals, vecs = eigenstates(transmon(), k=3)
    np.testing.assert_allclose(vals, eigenenergies(transmon(), k=3), atol=1e-9)
    # eigenvectors are normalised and solve H|v> = E|v>
    H = build_hamiltonian(transmon())
    for i in range(3):
        v = vecs[:, i]
        np.testing.assert_allclose(np.vdot(v, v), 1.0, atol=1e-9)
        np.testing.assert_allclose(H @ v, vals[i] * v, atol=1e-6)


def test_transmon_spectrum_converged_in_cutoff():
    # The low-lying transmon spectrum must be stable as the charge cutoff grows.
    e_small = eigenenergies(transmon(), k=3, charge_cut=12, relative=True)
    e_large = eigenenergies(transmon(), k=3, charge_cut=20, relative=True)
    np.testing.assert_allclose(e_small, e_large, atol=1e-8)


def test_fluxonium_uses_displacement_path_and_converges():
    # A cos(φ) on an extended (oscillator) mode goes through the BCH
    # displacement operator; check the low spectrum and its Fock convergence.
    e20 = eigenenergies(fluxonium(), k=3, fock_dim=20, relative=True)
    e30 = eigenenergies(fluxonium(), k=3, fock_dim=30, relative=True)
    np.testing.assert_allclose(e20, e30, atol=1e-4)
    np.testing.assert_allclose(e30, [0.0, 3.557675, 6.922798], atol=1e-5)
