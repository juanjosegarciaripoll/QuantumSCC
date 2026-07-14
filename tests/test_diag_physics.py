"""
Physics validation of the QuantumSCC diagonaliser (QuantumSCC.diag).

Complements test_diag.py with:
  * electromagnetic duality (transmon <-> dual-transmon spectra);
  * harmonic normal-mode reconstruction (coupled LC);
  * transmon anharmonicity sign;
  * cutoff / Fock-dimension convergence;
  * relabelling / element-ordering invariance of the spectrum;
  * regression snapshots pinning the low spectra of representative circuits.
"""

import numpy as np
import pytest

from QuantumSCC import Capacitor, Circuit, Inductor, Junction, PhaseSlip
from QuantumSCC.diag import build_hamiltonian, eigenenergies

# ── circuit factories ───────────────────────────────────────────────────────

def transmon(EJ=10.0, EC=0.2):
    return Circuit([(0, 1, Junction(EJ, "GHz")), (0, 1, Capacitor(EC, "GHz"))])


def dual_transmon(EP=10.0, EL=0.2):
    return Circuit([(0, 1, PhaseSlip(EP, "GHz")), (0, 1, Inductor(EL, "GHz"))])


def lc(EL=2.0, EC=0.5):
    return Circuit([(0, 1, Inductor(EL, "GHz")), (0, 1, Capacitor(EC, "GHz"))])


def coupled_lc():
    return Circuit([(0, 1, Inductor(1, "GHz")), (0, 1, Capacitor(1, "GHz")),
                    (0, 2, Inductor(1, "GHz")), (0, 2, Capacitor(1, "GHz"))])


def fluxonium():
    return Circuit([(0, 1, Junction(5, "GHz")), (0, 1, Inductor(1, "GHz")),
                    (0, 1, Capacitor(1, "GHz"))])


# ── electromagnetic duality ─────────────────────────────────────────────────

DUALITY_PARAMS = [
    (10.0, 0.2),
    (5.0, 0.5),
    (8.0, 1.0),
    (3.0, 0.3),
    (12.0, 0.25),
]


@pytest.mark.parametrize("EJ,EC", DUALITY_PARAMS,
                         ids=[f"EJ{a}_EC{b}" for a, b in DUALITY_PARAMS])
def test_transmon_dual_transmon_spectra_match(EJ, EC):
    """JJ<->QPS duality: transmon(E_J,E_C) and dual-transmon(E_P=E_J,E_L=E_C)
    must have identical spectra."""
    e_tr = eigenenergies(transmon(EJ, EC), k=5, relative=True)
    e_du = eigenenergies(dual_transmon(EJ, EC), k=5, relative=True)
    np.testing.assert_allclose(e_tr, e_du, atol=1e-6)


# ── harmonic normal-mode reconstruction ─────────────────────────────────────

def test_lc_ladder_matches_mode_frequency():
    """LC spectrum is an evenly spaced ladder at the mode frequency ω,
    which equals the (isotropic) K_flux diagonal entry."""
    c = lc()
    freq = float(np.diag(c.model().K_flux)[0])
    e = eigenenergies(c, k=5, relative=True)
    np.testing.assert_allclose(np.diff(e), [freq] * 4, atol=1e-6)


def test_coupled_lc_reproduces_normal_modes():
    """The two normal-mode frequencies (diag of K_flux, isotropic form) must
    appear as the first excitation energies of the coupled-LC spectrum."""
    c = coupled_lc()
    m = c.model()
    assert np.allclose(np.diag(m.K_flux), np.diag(m.K_charge))
    freqs = np.sort(np.diag(m.K_flux))
    e = eigenenergies(c, k=6, relative=True, fock_dim=10)
    # lowest excitation equals the smallest mode frequency
    np.testing.assert_allclose(e[1], freqs[0], atol=1e-4)
    # both mode frequencies appear in the low spectrum
    for f in freqs:
        assert np.any(np.isclose(e, f, atol=1e-4)), f"mode {f} missing from {e}"


# ── transmon anharmonicity ──────────────────────────────────────────────────

def test_transmon_negative_anharmonicity():
    """Transmon is a weakly anharmonic oscillator: the 1->2 gap is smaller
    than the 0->1 gap (negative anharmonicity)."""
    e = eigenenergies(transmon(), k=3, relative=True)
    gap01 = e[1] - e[0]
    gap12 = e[2] - e[1]
    anharm = gap12 - gap01
    assert anharm < 0.0
    # order of magnitude ~ -E_C (0.2 GHz); loose bound.
    assert -1.0 < anharm < -1e-3


# ── convergence ─────────────────────────────────────────────────────────────

def test_transmon_charge_cut_convergence():
    e12 = eigenenergies(transmon(), k=4, charge_cut=12, relative=True)
    e24 = eigenenergies(transmon(), k=4, charge_cut=24, relative=True)
    np.testing.assert_allclose(e12, e24, atol=1e-8)


def test_fluxonium_fock_dim_convergence():
    e20 = eigenenergies(fluxonium(), k=4, fock_dim=20, relative=True)
    e34 = eigenenergies(fluxonium(), k=4, fock_dim=34, relative=True)
    np.testing.assert_allclose(e20, e34, atol=1e-4)


# ── invariance ──────────────────────────────────────────────────────────────

def test_spectrum_invariant_under_node_relabeling():
    """Relabelling circuit nodes must not change the spectrum."""
    base = eigenenergies(transmon(), k=5, relative=True)
    relabelled = Circuit([(7, 3, Junction(10, "GHz")),
                          (7, 3, Capacitor(0.2, "GHz"))])
    np.testing.assert_allclose(base, eigenenergies(relabelled, k=5, relative=True),
                               atol=1e-6)


def test_spectrum_invariant_under_element_order():
    """Reordering the element list must not change the spectrum."""
    base = eigenenergies(fluxonium(), k=4, relative=True)
    reordered = Circuit([(0, 1, Capacitor(1, "GHz")),
                        (0, 1, Inductor(1, "GHz")),
                        (0, 1, Junction(5, "GHz"))])
    np.testing.assert_allclose(base, eigenenergies(reordered, k=4, relative=True),
                               atol=1e-4)


# ── Hermiticity / real spectrum ─────────────────────────────────────────────

@pytest.mark.parametrize("factory", [transmon, dual_transmon, lc, fluxonium,
                                     coupled_lc],
                         ids=["transmon", "dual_transmon", "lc", "fluxonium",
                              "coupled_lc"])
def test_hamiltonian_hermitian_and_real_spectrum(factory):
    H = build_hamiltonian(factory())
    assert abs((H - H.conj().T).toarray()).max() < 1e-10
    e = eigenenergies(factory(), k=4)
    assert np.all(np.isfinite(e))
    np.testing.assert_allclose(e.imag if np.iscomplexobj(e) else 0.0, 0.0, atol=1e-9)


# ── regression snapshots ────────────────────────────────────────────────────

# Pinned low spectra (relative, GHz) at charge_cut=14, fock_dim=16.
# Guards against silent numerical drift on refactors.
_SNAPSHOTS = {
    "lc": (lc, [0.0, 2.0, 4.0, 6.0, 8.0]),
    "transmon": (transmon, [0.0, 1.94866, 3.844137, 5.683943, 7.465157]),
    "fluxonium": (fluxonium, [0.0, 3.557666, 6.922843, 10.083319, 13.03031]),
    "dual_transmon": (lambda: dual_transmon(5.0, 1.0),
                      [0.0, 2.888705, 5.413587, 7.82959, 8.760403]),
    "coupled_lc": (coupled_lc, [0.0, 2.0, 2.0, 4.0, 4.0]),
}


@pytest.mark.parametrize("name", list(_SNAPSHOTS))
def test_spectrum_regression_snapshot(name):
    factory, expected = _SNAPSHOTS[name]
    e = eigenenergies(factory(), k=5, relative=True, charge_cut=14, fock_dim=16)
    np.testing.assert_allclose(e, expected, atol=1e-4)
