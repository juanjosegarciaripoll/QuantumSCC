"""
Unit tests for QuantumSCC.core.quantization (and supporting layers).

Structure
---------
1. CIRCUIT REGISTRY + INVARIANT TESTS (pytest parametrize)
   Mathematical properties that must hold for *any* valid circuit.
   Adding a new topology to CIRCUIT_REGISTRY automatically tests all invariants —
   no new test class or method needed.

2. PHYSICAL VALUE TESTS (unittest.TestCase)
   Specific analytical formulas that must match theory for concrete circuits.
   Examples: LC frequency = 1/sqrt(LC), dual-transmon H[0,0] = 2·E_L.

3. REGRESSION TESTS (unittest.TestCase)
   Specific nCF / nCC values that encode behaviour fixed by past bugs.
   These ensure the exact fix is preserved, not just that things "work".
"""

import pytest
import unittest
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC.core.elements import Capacitor, Inductor, Junction, PhaseSlip
from QuantumSCC.core.topology import Topology
from QuantumSCC.core.geometry import Geometry
from QuantumSCC.core.quantization import Quantization
from QuantumSCC.circuit import Circuit


# ══════════════════════════════════════════════════════════════════════════════
# 1. CIRCUIT REGISTRY + INVARIANT TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _J():
    return Junction(value=1, unit='GHz', cap=Capacitor(value=1, unit='GHz'))

def _P():
    return PhaseSlip(value=1, unit='GHz', ind=Inductor(value=1, unit='GHz'))

def _C():
    return Capacitor(value=1, unit='GHz')

def _L():
    return Inductor(value=1, unit='GHz')


# Each entry: (name, factory).  factory() returns a fresh list of (n1, n2, element).
# ─────────────────────────────────────────────────────────────────────────────
# Topologies covered:
#   Linear         : LC, coupled oscillators
#   Compact flux   : transmon (JJ), fluxonium (JJ+L), N JJ parallel/series
#   Compact charge : dual-transmon (QPS), dual-fluxonium (QPS+C), N QPS parallel/series
#   Mixed JJ+QPS   : same nodes (JJ‖QPS), chain (JJ-QPS), multi-node with shared nodes
#   Rings          : 3-node ring with JJ+QPS
# ─────────────────────────────────────────────────────────────────────────────
CIRCUIT_REGISTRY = [
    ("LC",
     lambda: [(0, 1, _L()), (0, 1, _C())]),

    ("coupled_LC",
     lambda: [(0, 1, _L()), (0, 1, _C()), (0, 2, _L()), (0, 2, _C())]),

    ("transmon",
     lambda: [(0, 1, _J())]),

    ("fluxonium",
     lambda: [(0, 1, _J()), (0, 1, _L())]),

    ("2JJ_parallel",
     lambda: [(0, 1, _J()), (0, 1, _J())]),

    ("2JJ_series",
     lambda: [(0, 1, _J()), (1, 2, _J()), (0, 2, _C())]),

    ("dual_transmon",
     lambda: [(0, 1, _P())]),

    ("dual_fluxonium",
     lambda: [(0, 1, _P()), (0, 1, _C())]),

    ("2QPS_parallel",
     lambda: [(0, 1, _P()), (0, 1, _P())]),

    ("2QPS_series",
     lambda: [(0, 1, _P()), (1, 2, _P()), (0, 2, _L())]),

    ("JJ_QPS_same_nodes",
     lambda: [(0, 1, _J()), (0, 1, _P())]),

    ("JJ_QPS_chain",
     lambda: [(0, 1, _J()), (1, 2, _P())]),

    ("2JJ_QPS_shared_node",
     lambda: [(0, 1, _J()), (1, 2, _J()), (0, 1, _P())]),

    ("JJ_QPS_JJ_chain",
     lambda: [(0, 1, _J()), (1, 2, _P()), (2, 3, _J())]),

    ("JJ_JJ_QPS_ring",
     lambda: [(0, 1, _J()), (1, 2, _J()), (2, 0, _P())]),

    ("QPS_QPS_JJ_ring",
     lambda: [(0, 1, _P()), (1, 2, _P()), (2, 0, _J())]),
]

_REGISTRY_IDS = [name for name, _ in CIRCUIT_REGISTRY]


@pytest.fixture(params=CIRCUIT_REGISTRY, ids=_REGISTRY_IDS)
def circuit(request):
    """Build a fresh Circuit for each entry in CIRCUIT_REGISTRY."""
    _, factory = request.param
    return Circuit(factory())


# ── Invariants ────────────────────────────────────────────────────────────────

def test_kirchhoff(circuit):
    """F · K = 0 (Kirchhoff constraint is exactly satisfied)."""
    assert np.allclose(circuit.topo.F @ circuit.topo.K, 0)


def test_hamiltonian_symmetric(circuit):
    """Quadratic Hamiltonian must be symmetric: H = Hᵀ."""
    H = circuit.quadratic_hamiltonian
    assert np.allclose(H, H.T)


def test_hamiltonian_real(circuit):
    """Quadratic Hamiltonian must be real (no imaginary part)."""
    assert np.allclose(circuit.quadratic_hamiltonian.imag, 0)


def test_symplectic_form_antisymmetric(circuit):
    """Symplectic form Ω must be anti-symmetric: Ω = −Ωᵀ."""
    O = circuit.geom.omega_symplectic
    assert np.allclose(O, -O.T)


def test_symplectic_rank(circuit):
    """rank(Ω) must equal no_independent_variables (Darboux reduction is non-degenerate)."""
    O = circuit.geom.omega_symplectic
    n = circuit.geom.no_independent_variables
    assert np.linalg.matrix_rank(O) == n


def test_compact_counts_bounded(circuit):
    """Compact variable counts cannot exceed the number of generating elements."""
    assert circuit.no_final_compact_flux   <= circuit.no_JJ
    assert circuit.no_final_compact_charge <= circuit.no_QPS


def test_jj_vector_in_dynamical_sector(circuit):
    """When JJ elements are present, vector_JJ must have a non-zero dynamical component."""
    if circuit.no_JJ > 0:
        assert not np.allclose(circuit.vector_JJ, 0)


def test_qps_vector_in_dynamical_sector(circuit):
    """When QPS elements are present, vector_QPS must have a non-zero dynamical component."""
    if circuit.no_QPS > 0:
        assert not np.allclose(circuit.vector_QPS, 0)


def test_harmonic_frequencies_nonnegative(circuit):
    """All harmonic-mode frequencies (diagonal of extended_quantum_hamiltonian) must be ≥ 0."""
    H = circuit.extended_quantum_hamiltonian
    if H.shape[0] == 0:
        return  # no harmonic modes → trivially satisfied
    n = H.shape[0] // 2
    freqs = [H[i, i].real for i in range(n)]
    assert all(f >= -1e-10 for f in freqs)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PHYSICAL VALUE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestLCFrequency(unittest.TestCase):
    """LC oscillator frequency must match ω = 1/√(LC)."""

    def setUp(self):
        self.C = Capacitor(value=1, unit='pF')
        self.L = Inductor(value=1, unit='nH')
        topo = Topology([(0, 1, self.L), (0, 1, self.C)])
        geom = Geometry(topo)
        self.quant = Quantization(topo, geom)

    def test_frequency_matches_formula(self):
        val_c = self.C.cValue * 1e-12
        val_l = self.L.lValue * 1e-9
        omega_theory = 1e-9 / np.sqrt(val_c * val_l)
        H = self.quant.extended_quantum_hamiltonian.real
        self.assertTrue(np.allclose(np.diag(H), omega_theory),
                        msg=f"Got {np.diag(H)}, expected {omega_theory:.4f}")

    def test_hamiltonian_positive_definite(self):
        H = self.quant.quadratic_hamiltonian
        eigvals = np.linalg.eigvals(H)
        self.assertTrue(np.all(eigvals > 0))


class TestDualTransmonValues(unittest.TestCase):
    """
    Dual-transmon: QPS(1 GHz) ∥ Inductor(1 nH).
    H = E_L φ² − E_P cos(q).
    Expected: H[0,0] = 2·E_L_code, H[1,1] = 0 (compact charge has no quadratic term).
    """

    def setUp(self):
        from QuantumSCC.utils import units as unt
        self.L_nH = 1.0
        L = Inductor(value=self.L_nH, unit='nH')
        P = PhaseSlip(value=1.0, unit='GHz', ind=L)
        topo  = Topology([(0, 1, P)])
        geom  = Geometry(topo)
        self.quant = Quantization(topo, geom)
        self.E_L_code = (unt.Phi0 / (2 * np.pi))**2 / (
            2 * self.L_nH * 1e-9 * unt.hbar) / 1e9

    def test_flux_entry_equals_2_E_L(self):
        H = self.quant.quadratic_hamiltonian
        expected = 2.0 * self.E_L_code
        self.assertAlmostEqual(H[0, 0].real, expected, delta=expected * 1e-6)

    def test_charge_entry_zero(self):
        H = self.quant.quadratic_hamiltonian
        self.assertAlmostEqual(abs(H[1, 1]), 0.0, delta=1e-10)

    def test_vector_qps_single_column(self):
        self.assertEqual(self.quant.vector_QPS.shape[1], 1)

    def test_vector_jj_empty(self):
        self.assertEqual(self.quant.vector_JJ.shape[1], 0)


class TestDualFluxoniumValues(unittest.TestCase):
    """
    Dual-fluxonium: QPS ∥ C.
    The parallel capacitor extends the QPS charge (kills compact mode).
    Expected: nCC = 0, nCF = 0, exactly 1 harmonic mode, H[0,0] > 0, H[1,1] > 0.
    """

    def setUp(self):
        L = Inductor(value=1.0, unit='nH')
        P = PhaseSlip(value=5.0, unit='GHz', ind=L)
        C = Capacitor(value=1.0, unit='GHz')
        self.topo  = Topology([(0, 1, P), (0, 1, C)])
        self.geom  = Geometry(self.topo)
        self.quant = Quantization(self.topo, self.geom)

    def test_capacitor_kills_compact_charge(self):
        self.assertEqual(self.geom.no_final_compact_charge, 0)

    def test_no_compact_flux(self):
        self.assertEqual(self.geom.no_final_compact_flux, 0)

    def test_exactly_one_harmonic_mode(self):
        n_osc = (self.geom.no_independent_variables // 2
                 - self.geom.no_final_compact_flux
                 - self.geom.no_final_compact_charge)
        self.assertEqual(n_osc, 1)

    def test_hamiltonian_entries_positive(self):
        H = self.quant.quadratic_hamiltonian
        self.assertGreater(H[0, 0].real, 0)
        self.assertGreater(H[1, 1].real, 0)

    def test_kirchhoff(self):
        self.assertTrue(np.allclose(self.topo.F @ self.topo.K, 0))


class TestParallelQPSScaling(unittest.TestCase):
    """
    N identical QPS in parallel on the same node pair.
    Physical scaling laws:
      - Still 1 compact charge mode regardless of N.
      - Parallel inductors halve/third the effective L → inductive energy ∝ N.
      - Identical QPS share equal coupling vectors.
    """

    def _build(self, n_qps, el=1.0, ep=0.5):
        inds = [Inductor(el, 'GHz') for _ in range(n_qps)]
        qps  = [PhaseSlip(ep, 'GHz', ind=inds[k]) for k in range(n_qps)]
        topo  = Topology([(0, 1, qps[k]) for k in range(n_qps)])
        geom  = Geometry(topo)
        quant = Quantization(topo, geom)
        return topo, geom, quant

    def test_always_one_compact_charge_mode(self):
        for n in (1, 2, 3, 4):
            topo, _, _ = self._build(n)
            self.assertEqual(topo.no_reduced_compact_charge, 1,
                             msg=f"N={n} QPS parallel: expected nCC=1")

    def test_inductive_energy_scales_with_n(self):
        _, _, q1 = self._build(1, el=1.0)
        _, _, q2 = self._build(2, el=1.0)
        _, _, q3 = self._build(3, el=1.0)
        h1 = q1.quadratic_hamiltonian[0, 0]
        self.assertAlmostEqual(q2.quadratic_hamiltonian[0, 0], 2 * h1, places=10)
        self.assertAlmostEqual(q3.quadratic_hamiltonian[0, 0], 3 * h1, places=10)

    def test_identical_parallel_qps_share_equal_vectors(self):
        for n in (2, 3):
            _, _, quant = self._build(n)
            v = quant.vector_QPS
            for k in range(1, n):
                self.assertTrue(np.allclose(v[:, 0], v[:, k]),
                                msg=f"N={n}: QPS #{k} vector differs from QPS #0")

    def test_qps_groups_one_group_per_node_pair(self):
        topo1, _, _ = self._build(1)
        topo2, _, _ = self._build(2)
        self.assertEqual(len(topo1.qps_groups), 1)
        self.assertEqual(len(topo2.qps_groups), 1)
        self.assertEqual(len(list(topo2.qps_groups.values())[0]), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 3. REGRESSION TESTS
# Each test pins the exact behaviour introduced by a specific bug fix.
# ══════════════════════════════════════════════════════════════════════════════

class TestQPSRegressions(unittest.TestCase):
    """
    Regression tests for QPS bugs fixed on branch feature/refactoring.

    Bug A — JJ ∥ QPS same nodes
        The QPS inductor makes the JJ flux extended (nCF = 0).
        Previously raised ValueError; now must build and give nCF=0, nCC=1.

    Bug B — JJ-QPS chain (crossed pairing)
        compact-flux conjugates with extended-charge (not compact-charge).
        Previously raised ValueError; now nCF=1, nCC=1.

    Bug C — nCF>0 and nEF>0 simultaneously (Block 3 null-space scope)
        2JJ + QPS sharing one node pair: previously 'QPS fully decoupled'.
        Now nCF=1, nCC=1.

    Bug D — Ring topologies (QPS Kirchhoff projection)
        Relaxed validation: non-zero non-dynamical components are physical.
        Rings now build with correct compact variable counts.
    """

    # ── Bug A: JJ ∥ QPS same nodes ───────────────────────────────────────────

    def test_jj_qps_same_nodes_builds(self):
        Circuit([(0, 1, _J()), (0, 1, _P())])  # must not raise

    def test_jj_qps_same_nodes_nCF_zero(self):
        """QPS inductor makes JJ flux extended → nCF = 0."""
        c = Circuit([(0, 1, _J()), (0, 1, _P())])
        self.assertEqual(c.topo.no_reduced_compact_flux, 0)

    def test_jj_qps_same_nodes_nCC_one(self):
        c = Circuit([(0, 1, _J()), (0, 1, _P())])
        self.assertEqual(c.topo.no_reduced_compact_charge, 1)

    # ── Bug B: JJ-QPS chain ──────────────────────────────────────────────────

    def test_jj_qps_chain_builds(self):
        Circuit([(0, 1, _J()), (1, 2, _P())])  # must not raise

    def test_jj_qps_chain_nCF_one(self):
        c = Circuit([(0, 1, _J()), (1, 2, _P())])
        self.assertEqual(c.topo.no_reduced_compact_flux, 1)

    def test_jj_qps_chain_nCC_one(self):
        c = Circuit([(0, 1, _J()), (1, 2, _P())])
        self.assertEqual(c.topo.no_reduced_compact_charge, 1)

    # ── Bug C: 2JJ + QPS with shared node ────────────────────────────────────

    def test_2jj_qps_shared_node_builds(self):
        Circuit([(0, 1, _J()), (1, 2, _J()), (0, 1, _P())])

    def test_2jj_qps_shared_node_nCC_one(self):
        """QPS charge must land in the dynamical sector (Block 3 fix)."""
        c = Circuit([(0, 1, _J()), (1, 2, _J()), (0, 1, _P())])
        self.assertFalse(np.allclose(c.vector_QPS, 0))
        self.assertEqual(c.topo.no_reduced_compact_charge, 1)

    # ── Bug D: Ring topologies ────────────────────────────────────────────────

    def test_jj_jj_qps_ring_builds(self):
        Circuit([(0, 1, _J()), (1, 2, _J()), (2, 0, _P())])

    def test_qps_qps_jj_ring_builds(self):
        Circuit([(0, 1, _P()), (1, 2, _P()), (2, 0, _J())])

    def test_jj_jj_qps_ring_nCC_one(self):
        c = Circuit([(0, 1, _J()), (1, 2, _J()), (2, 0, _P())])
        self.assertFalse(np.allclose(c.vector_QPS, 0))
        self.assertEqual(c.topo.no_reduced_compact_charge, 1)

    def test_qps_qps_jj_ring_compact_counts(self):
        """QPS-QPS-JJ ring: two QPS inductors extend the JJ flux → nCF=0, nCC=2."""
        c = Circuit([(0, 1, _P()), (1, 2, _P()), (2, 0, _J())])
        self.assertFalse(np.allclose(c.vector_JJ, 0))
        self.assertEqual(c.topo.no_reduced_compact_flux, 0)
        self.assertEqual(c.topo.no_reduced_compact_charge, 2)


if __name__ == '__main__':
    unittest.main()
