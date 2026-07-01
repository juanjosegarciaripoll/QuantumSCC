"""
Registry-wide property tests (Tier E).

Every topology in conftest.CIRCUIT_REGISTRY must satisfy the same structural
invariants of the QuantumSCC pipeline, and (where the diagonaliser supports it)
produce a Hermitian Hamiltonian with a real, finite spectrum. Parametrising one
set of checks over the whole registry guards every supported circuit uniformly,
rather than ad hoc per-architecture.
"""

import numpy as np
import pytest
from conftest import CIRCUIT_REGISTRY

from QuantumSCC import Circuit
from QuantumSCC.diag import build_hamiltonian, eigenenergies

# Small Hilbert-space cutoffs keep the registry sweep fast.
_CHARGE_CUT = 3
_FOCK_DIM = 4

# dualmon_full's compact-charge quadratic term references a periodic operator,
# which the diagonaliser cannot reconstruct ("periodic operators only in
# cosines"). Its structural invariants still hold; only diagonalisation is
# excluded (and pinned separately below).
_NON_DIAGONALISABLE = {"dualmon_full"}
_DIAG_REGISTRY = [(n, f) for n, f in CIRCUIT_REGISTRY
                  if n not in _NON_DIAGONALISABLE]

_ALL_IDS = [n for n, _ in CIRCUIT_REGISTRY]
_DIAG_IDS = [n for n, _ in _DIAG_REGISTRY]


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


# ── diagonalisation invariants (reconstructible circuits) ───────────────────

@pytest.mark.parametrize("factory", [f for _, f in _DIAG_REGISTRY], ids=_DIAG_IDS)
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


# ── documented non-reconstructible circuit ──────────────────────────────────

def test_dualmon_full_periodic_quadratic_not_reconstructible():
    """dualmon_full has a compact-charge quadratic term on a periodic operator;
    the diagonaliser must reject it with a clear error rather than build a wrong
    Hamiltonian."""
    factory = dict(CIRCUIT_REGISTRY)["dualmon_full"]
    with pytest.raises(ValueError, match="periodic operator"):
        build_hamiltonian(Circuit(factory()),
                          charge_cut=_CHARGE_CUT, fock_dim=_FOCK_DIM)
