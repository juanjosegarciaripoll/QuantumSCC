"""
Registry-wide property tests (Tier E).

Every topology in conftest.CIRCUIT_REGISTRY must satisfy the same structural
invariants of the QuantumSCC pipeline, and produce a Hermitian Hamiltonian with
a real, finite spectrum. Parametrising one
set of checks over the whole registry guards every supported circuit uniformly,
rather than ad hoc per-architecture.
"""

import dataclasses

import numpy as np
import pytest
from conftest import CIRCUIT_REGISTRY

from QuantumSCC import Circuit
from QuantumSCC.diag import build_hamiltonian, eigenenergies

# Small Hilbert-space cutoffs keep the registry sweep fast.
_CHARGE_CUT = 3
_FOCK_DIM = 4

_ALL_IDS = [n for n, _ in CIRCUIT_REGISTRY]


def _canonical_J(k):
    return np.block([[np.zeros((k, k)), np.eye(k)],
                     [-np.eye(k), np.zeros((k, k))]])


# ── structural invariants (all circuits) ────────────────────────────────────

@pytest.mark.parametrize("factory", [f for _, f in CIRCUIT_REGISTRY], ids=_ALL_IDS)
class TestStructuralInvariants:
    def test_kirchhoff_fk_zero(self, factory):
        c = Circuit(factory())
        assert np.allclose(c.topo.F @ c.topo.K, 0, atol=1e-10)

    def test_quadratic_hamiltonian_symmetric_real_psd(self, factory):
        H = Circuit(factory()).quadratic_hamiltonian
        np.testing.assert_allclose(H.imag, 0, atol=1e-12)
        Hr = H.real
        np.testing.assert_allclose(Hr, Hr.T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh((Hr + Hr.T) / 2) >= -1e-9)

    def test_omega_2b_antisymmetric(self, factory):
        Om = Circuit(factory()).omega_2B
        np.testing.assert_allclose(Om, -Om.T, atol=1e-12)

    def test_omega_symplectic_canonical(self, factory):
        Om = Circuit(factory()).omega_symplectic.real
        n = Om.shape[0]
        assert n % 2 == 0
        np.testing.assert_allclose(Om, _canonical_J(n // 2), atol=1e-9)

    def test_compact_counts_bounded(self, factory):
        c = Circuit(factory())
        assert c.no_final_compact_flux <= c.topo.no_JJ
        assert c.no_final_compact_charge <= c.topo.no_QPS

    def test_independent_variables_even(self, factory):
        assert Circuit(factory()).no_independent_variables % 2 == 0

    def test_basis_change_square(self, factory):
        V = Circuit(factory()).V
        assert V.shape[0] == V.shape[1]

    def test_mode_count_consistency(self, factory):
        c = Circuit(factory())
        n = c.no_independent_variables // 2
        assert len(c.conjugate_pairs()) == n
        assert c.model().n_modes == n


# ── diagonalisation invariants (all circuits) ───────────────────────────────

@pytest.mark.parametrize("factory", [f for _, f in CIRCUIT_REGISTRY], ids=_ALL_IDS)
class TestDiagonalisationInvariants:
    def test_hamiltonian_hermitian(self, factory):
        H = build_hamiltonian(Circuit(factory()),
                              charge_cut=_CHARGE_CUT, fock_dim=_FOCK_DIM)
        assert abs((H - H.conj().T).toarray()).max() < 1e-10

    def test_spectrum_real_and_finite(self, factory):
        e = eigenenergies(Circuit(factory()), k=2,
                          charge_cut=_CHARGE_CUT, fock_dim=_FOCK_DIM)
        assert np.all(np.isfinite(e))
        # eigenenergies returns real values; ordering is ascending.
        assert e[0] <= e[1] + 1e-9

    def test_circuit_and_model_give_same_hamiltonian(self, factory):
        c = Circuit(factory())
        h1 = build_hamiltonian(c, charge_cut=_CHARGE_CUT, fock_dim=_FOCK_DIM)
        h2 = build_hamiltonian(c.model(), charge_cut=_CHARGE_CUT, fock_dim=_FOCK_DIM)
        assert abs((h1 - h2).toarray()).max() < 1e-12


# ── dualmon_full: mixed compact-charge + extended circuit ───────────────────

def _dualmon_full_reference(N=121, X=12.0, M=6, k=5):
    """Lowest ``k`` levels of dualmon_full, built independently of QuantumSCC.

    Circuit: JJ ‖ C ‖ C on (1,0), L on (1,2), QPS ‖ L on (2,0), all values 1 GHz.
    With φ₁ the node-1 phase, n₁ its conjugate charge, q the (compact) QPS charge
    and ψ ∈ ℤ its conjugate flux:

        H = ½ n₁² + (φ₁ − ψ)² + ψ² − cos φ₁ − cos q

    (two parallel ``Capacitor(1)`` → ½ n₁², each ``Inductor(1)`` → (Δφ)²).
    φ₁ is discretised on a sinc-DVR grid of ``N`` points on [−X, X]; ψ ∈ [−M, M].
    """
    x = np.linspace(-X, X, N)
    h = x[1] - x[0]
    d = np.arange(N)[:, None] - np.arange(N)[None, :]
    with np.errstate(divide="ignore"):
        kinetic = np.where(d == 0, np.pi**2 / 3, 2.0 * (-1.0) ** d / d**2) / h**2
    psis = np.arange(-M, M + 1)
    H = np.zeros((len(psis) * N, len(psis) * N))
    for a, psi in enumerate(psis):
        blk = slice(a * N, (a + 1) * N)
        H[blk, blk] = 0.5 * kinetic + np.diag((x - psi) ** 2 + psi**2 - np.cos(x))
    for a in range(len(psis) - 1):  # − cos q = −½ (ψ → ψ+1 + h.c.)
        lo, hi = slice(a * N, (a + 1) * N), slice((a + 1) * N, (a + 2) * N)
        H[lo, hi] = H[hi, lo] = -0.5 * np.eye(N)
    return np.linalg.eigvalsh(H)[:k]


def test_dualmon_full_matches_independent_reference():
    """The compact QPS charge enters only through cos q, so dualmon_full is
    reconstructible and its spectrum matches the hand-built Hamiltonian."""
    factory = dict(CIRCUIT_REGISTRY)["dualmon_full"]
    e = eigenenergies(Circuit(factory()), k=5, charge_cut=8, fock_dim=30)
    np.testing.assert_allclose(e, _dualmon_full_reference(), atol=1e-7)


def test_periodic_quadratic_term_is_rejected():
    """A quadratic term on a periodic operator violates the model invariant;
    the diagonaliser must reject it rather than build a wrong Hamiltonian."""
    model = Circuit(dict(CIRCUIT_REGISTRY)["dualmon_full"]()).model()
    assert model.mode_sectors[0] == "compact_charge"
    K_charge = np.array(model.K_charge, dtype=float)
    K_charge[0, 0] = 1.0
    with pytest.raises(ValueError, match="periodic operator"):
        build_hamiltonian(dataclasses.replace(model, K_charge=K_charge),
                          charge_cut=_CHARGE_CUT, fock_dim=_FOCK_DIM)
