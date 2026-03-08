"""
Concrete-circuit tests for QuantumSCC (full pipeline).

All tests in this file use specific circuits with exact expected values.
No generic parametrized invariants live here — those are in test_topology.py,
test_geometry.py, and test_quantization.py.

Sections
--------
1. Physical value tests   — LC frequency, 4·E_C, 2·E_L, duality JJ↔QPS
2. Topology edge cases    — which elements create / kill compact modes
3. Geometry mode counts   — harmonic vs compact mode structure
4. QPS scaling laws       — N parallel QPS inductive energy ∝ N
5. QPS regression tests   — exact nCF / nCC values pinned by 5 bug fixes
6. Error handling         — unsupported configurations
7. Backwards compatibility — pre-QPS circuits unaffected
"""

import unittest
import numpy as np
import os
import sys
from io import StringIO
import contextlib

current_dir  = os.path.dirname(os.path.abspath(__file__))
package_dir  = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip
from QuantumSCC.core.topology import Topology
from QuantumSCC.core.geometry import Geometry
from QuantumSCC.core.quantization import Quantization
from QuantumSCC.utils import units as unt


# ── LC oscillator ─────────────────────────────────────────────────────────────

class TestCircuitLC(unittest.TestCase):
    """LC oscillator — simplest complete circuit (1 harmonic mode)."""

    def setUp(self):
        self.C = Capacitor(value=1, unit='pF')
        self.L = Inductor(value=1, unit='nH')
        self.circuit = Circuit([(0, 1, self.L), (0, 1, self.C)])

    # --- attribute delegation ---

    def test_attribute_delegation_topology(self):
        """Topological attributes on Circuit are the same objects as in Topology."""
        c = self.circuit
        self.assertIs(c.Fcut, c.topo.Fcut)
        self.assertIs(c.Floop, c.topo.Floop)
        self.assertIs(c.F, c.topo.F)
        self.assertIs(c.K, c.topo.K)

    def test_attribute_delegation_geometry(self):
        """Geometric attributes on Circuit are the same objects as in Geometry."""
        c = self.circuit
        self.assertIs(c.omega_2B,        c.geom.omega_2B)
        self.assertIs(c.omega_symplectic, c.geom.omega_symplectic)
        self.assertIs(c.V,               c.geom.V)

    def test_attribute_delegation_quantization(self):
        """Quantization attributes on Circuit are the same objects as in Quantization."""
        c = self.circuit
        self.assertIs(c.quadratic_hamiltonian,        c.quant.quadratic_hamiltonian)
        self.assertIs(c.extended_quantum_hamiltonian,  c.quant.extended_quantum_hamiltonian)
        self.assertIs(c.vector_JJ,                    c.quant.vector_JJ)

    # --- element counts ---

    def test_element_counts(self):
        self.assertEqual(self.circuit.no_JJ,         0)
        self.assertEqual(self.circuit.no_Capacitors,  1)
        self.assertEqual(self.circuit.no_Inductors,   1)
        self.assertEqual(self.circuit.no_elements,    2)

    # --- modes ---

    def test_one_dynamic_mode(self):
        """LC oscillator has exactly 1 dynamic mode (2 independent variables)."""
        self.assertEqual(self.circuit.no_independent_variables, 2)

    def test_no_junctions_vector_empty(self):
        """With no junctions, vector_JJ has 0 columns."""
        self.assertEqual(self.circuit.vector_JJ.shape[1], 0)

    # --- physical values ---

    def test_frequency_matches_formula(self):
        """Mode frequency must equal ω = 1/√(L·C) (in GHz)."""
        val_c = self.C.cValue * 1e-12
        val_l = self.L.lValue * 1e-9
        omega_theory = 1e-9 / np.sqrt(val_c * val_l)
        H = self.circuit.extended_quantum_hamiltonian.real
        self.assertTrue(np.allclose(np.diag(H), omega_theory),
                        msg=f"Expected {omega_theory:.4f}, got {np.diag(H)}")

    def test_hamiltonian_positive_definite(self):
        """All eigenvalues of the quadratic Hamiltonian must be positive."""
        H = self.circuit.quadratic_hamiltonian
        eigvals = np.linalg.eigvals(H)
        self.assertTrue(np.all(eigvals.real > 0))

    # --- display ---

    def test_diagonal_hamiltonian_display(self):
        """diagonal_harmonic_Hamiltonian_expression() must produce non-empty output."""
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            self.circuit.diagonal_harmonic_Hamiltonian_expression()
        self.assertGreater(len(buf.getvalue()), 0)


# ── Two independent LC oscillators ────────────────────────────────────────────

class TestCircuitTwoLC(unittest.TestCase):
    """Two independent LC oscillators sharing a common ground (2 modes, no coupling)."""

    def setUp(self):
        C = Capacitor(value=1, unit='pF')
        L = Inductor(value=1, unit='nH')
        self.circuit = Circuit([
            (0, 1, L), (0, 1, C),
            (0, 2, L), (0, 2, C),
        ])
        self.omega = 1e-9 / np.sqrt(1e-12 * 1e-9)

    def test_two_dynamic_modes(self):
        """Two LC loops → 4 independent variables (2 modes)."""
        self.assertEqual(self.circuit.no_independent_variables, 4)

    def test_hamiltonian_shape(self):
        """Quadratic Hamiltonian must be 4×4 for a 2-mode circuit."""
        self.assertEqual(self.circuit.quadratic_hamiltonian.shape, (4, 4))

    def test_hamiltonian_symmetric(self):
        H = self.circuit.quadratic_hamiltonian
        self.assertTrue(np.allclose(H, H.T))

    def test_both_frequencies_positive(self):
        """Both mode frequencies must be strictly positive."""
        H = self.circuit.extended_quantum_hamiltonian.real
        n = H.shape[0] // 2
        freqs = [H[i, i] for i in range(n)]
        self.assertTrue(all(f > 0 for f in freqs),
                        msg=f"Non-positive frequency found: {freqs}")


# ── Fluxonium ─────────────────────────────────────────────────────────────────

class TestCircuitFluxonium(unittest.TestCase):
    """Fluxonium (JJ ∥ L) — nonlinear circuit with 1 harmonic mode."""

    def setUp(self):
        C_J = Capacitor(value=1, unit='pF')
        J   = Junction(value=1, unit='GHz', cap=C_J)
        L   = Inductor(value=1, unit='nH')
        self.circuit = Circuit([(0, 1, J), (0, 1, L)])

    def test_element_counts(self):
        self.assertEqual(self.circuit.no_JJ,         1)
        self.assertEqual(self.circuit.no_Capacitors,  1)
        self.assertEqual(self.circuit.no_Inductors,   1)

    def test_jj_vector_nonzero(self):
        """JJ coupling vector must not be zero."""
        self.assertFalse(np.allclose(self.circuit.vector_JJ, 0))

    def test_jj_vector_single_column(self):
        """One junction → vector_JJ has exactly 1 column."""
        self.assertEqual(self.circuit.vector_JJ.shape[1], 1)

    def test_hamiltonian_phiq_square(self):
        """FS_quadratic_hamiltonian_phiq must be square."""
        H = self.circuit.FS_quadratic_hamiltonian_phiq
        self.assertEqual(H.shape[0], H.shape[1])

    def test_hamiltonian_expression_prints_cos(self):
        """Hamiltonian_expression() must include the cos(…) Josephson term."""
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            self.circuit.Hamiltonian_expression()
        self.assertIn('cos', buf.getvalue())


# ── Transmon (physical values) ────────────────────────────────────────────────

class TestCircuitTransmon(unittest.TestCase):
    """
    Transmon: JJ with parallel capacitor.
    H = 4·E_C·n² − E_J·cos(φ).
    Expected: H[0,0] = 0 (compact flux has no quadratic term), H[1,1] = 4·E_C.
    """

    def setUp(self):
        self.C_pF = 1.0
        C = Capacitor(value=self.C_pF, unit='pF')
        J = Junction(value=1, unit='GHz', cap=C)
        self.circuit = Circuit([(0, 1, J)])
        self.E_C_code = (2 * unt.e)**2 / (2 * self.C_pF * 1e-12 * unt.hbar) / 1e9

    def test_compact_flux_entry_zero(self):
        """H[0,0] = 0: compact flux has no quadratic term."""
        H = self.circuit.quadratic_hamiltonian
        self.assertAlmostEqual(abs(H[0, 0]), 0.0, delta=1e-10)

    def test_charge_entry_equals_2_EC(self):
        """H[1,1] = 2·E_C (integer kernel normalisation: K columns have norm √2)."""
        H = self.circuit.quadratic_hamiltonian
        self.assertAlmostEqual(H[1, 1].real, 2.0 * self.E_C_code,
                               delta=self.E_C_code * 1e-5)

    def test_compact_flux_one(self):
        self.assertEqual(self.circuit.no_final_compact_flux, 1)

    def test_compact_charge_zero(self):
        self.assertEqual(self.circuit.no_final_compact_charge, 0)


# ── Dual-transmon (physical values) ──────────────────────────────────────────

class TestCircuitDualTransmon(unittest.TestCase):
    """
    Dual-transmon: QPS(1 GHz) with parallel Inductor(1 nH).
    H = E_L·φ² − E_P·cos(q).
    Expected: H[0,0] = 2·E_L, H[1,1] = 0 (compact charge has no quadratic term).
    """

    def setUp(self):
        self.L_nH = 1.0
        P = PhaseSlip(value=1.0, unit='GHz', L_value=self.L_nH, L_unit='nH')
        self.circuit  = Circuit([(0, 1, P)])
        self.E_L_code = (unt.Phi0 / (2 * np.pi))**2 / (
            2 * self.L_nH * 1e-9 * unt.hbar) / 1e9

    def test_element_counts(self):
        self.assertEqual(self.circuit.no_QPS,        1)
        self.assertEqual(self.circuit.no_JJ,         0)
        self.assertEqual(self.circuit.no_Inductors,   1)
        self.assertEqual(self.circuit.no_Capacitors,  0)

    def test_compact_charge_count(self):
        self.assertEqual(self.circuit.no_final_compact_charge, 1)

    def test_compact_flux_count(self):
        self.assertEqual(self.circuit.no_final_compact_flux, 0)

    def test_flux_entry_equals_2_EL(self):
        """H[0,0] = 2·E_L_code — Article Eq. (30)."""
        H = self.circuit.quadratic_hamiltonian
        expected = 2.0 * self.E_L_code
        self.assertAlmostEqual(H[0, 0].real, expected, delta=expected * 1e-6)

    def test_charge_entry_zero(self):
        """H[1,1] = 0: compact charge has no quadratic term."""
        self.assertAlmostEqual(abs(self.circuit.quadratic_hamiltonian[1, 1]), 0.0, delta=1e-10)

    def test_vector_qps_nonzero_single_column(self):
        self.assertFalse(np.allclose(self.circuit.vector_QPS, 0))
        self.assertEqual(self.circuit.vector_QPS.shape[1], 1)

    def test_vector_jj_empty(self):
        self.assertEqual(self.circuit.vector_JJ.shape[1], 0)

    def test_hamiltonian_expression_prints_cos(self):
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            self.circuit.Hamiltonian_expression()
        self.assertIn('cos', buf.getvalue())


# ── JJ ↔ QPS duality ─────────────────────────────────────────────────────────

class TestQPSJJDuality(unittest.TestCase):
    """
    Structural duality check: the Hamiltonian is symmetric under JJ ↔ QPS.
      Transmon:       H[0,0]=0  (compact flux),  H[1,1]=2·E_C
      Dual-transmon:  H[1,1]=0  (compact charge), H[0,0]=4·E_L
    Integer kernel normalisation: compact K columns have norm √2,
    so compact-sector entries differ by factor 2 from SVD-normalised basis.
    Compact variable counts are swapped between the two.
    """

    def setUp(self):
        C = Capacitor(value=1, unit='pF')
        J = Junction(value=1, unit='GHz', cap=C)
        self.transmon    = Circuit([(0, 1, J)])
        self.E_C_code    = (2 * unt.e)**2 / (2 * 1e-12 * unt.hbar) / 1e9

        P = PhaseSlip(value=1, unit='GHz', L_value=1, L_unit='nH')
        self.dual        = Circuit([(0, 1, P)])
        self.E_L_code    = (unt.Phi0 / (2 * np.pi))**2 / (2 * 1e-9 * unt.hbar) / 1e9

    def test_transmon_compact_flux_entry_zero(self):
        self.assertAlmostEqual(abs(self.transmon.quadratic_hamiltonian[0, 0]), 0.0, delta=1e-10)

    def test_transmon_charge_entry_formula(self):
        """H[1,1] = 2·E_C (integer kernel normalisation)."""
        H = self.transmon.quadratic_hamiltonian
        self.assertAlmostEqual(H[1, 1].real, 2.0 * self.E_C_code,
                               delta=self.E_C_code * 1e-5)

    def test_dual_compact_charge_entry_zero(self):
        self.assertAlmostEqual(abs(self.dual.quadratic_hamiltonian[1, 1]), 0.0, delta=1e-10)

    def test_dual_flux_entry_formula(self):
        """H[0,0] = 2·E_L — Article Eq. (30)."""
        H = self.dual.quadratic_hamiltonian
        self.assertAlmostEqual(H[0, 0].real, 2.0 * self.E_L_code,
                               delta=self.E_L_code * 1e-5)

    def test_compact_variable_counts_swapped(self):
        self.assertEqual(self.transmon.no_final_compact_flux,    1)
        self.assertEqual(self.transmon.no_final_compact_charge,  0)
        self.assertEqual(self.dual.no_final_compact_flux,        0)
        self.assertEqual(self.dual.no_final_compact_charge,      1)


# ── Error handling ────────────────────────────────────────────────────────────

class TestCircuitErrors(unittest.TestCase):
    """Circuit-level error handling (cases that require building a Circuit)."""

    def test_jj_and_qps_in_parallel_builds(self):
        """JJ ∥ QPS on the same nodes: now supported — must not raise."""
        C = Capacitor(value=1, unit='pF')
        J = Junction(value=1, unit='GHz', cap=C)
        P = PhaseSlip(value=1, unit='GHz', L_value=1, L_unit='nH')
        circuit = Circuit([(0, 1, J), (0, 1, P)])
        self.assertIsNotNone(circuit.quadratic_hamiltonian)


# ── Backwards compatibility ───────────────────────────────────────────────────

class TestBackwardsCompatibility(unittest.TestCase):
    """Pre-QPS circuits must be completely unaffected by the QPS extension."""

    def test_lc_frequency_unchanged(self):
        C = Capacitor(value=1, unit='pF')
        L = Inductor(value=1, unit='nH')
        c = Circuit([(0, 1, L), (0, 1, C)])
        omega = 1e-9 / np.sqrt(1e-12 * 1e-9)
        self.assertTrue(np.allclose(np.diag(c.extended_quantum_hamiltonian.real), omega))

    def test_transmon_no_compact_charge(self):
        C = Capacitor(value=1, unit='pF')
        J = Junction(value=1, unit='GHz', cap=C)
        c = Circuit([(0, 1, J)])
        self.assertEqual(c.no_final_compact_charge, 0)
        self.assertEqual(c.no_QPS,                  0)

    def test_fluxonium_kirchhoff(self):
        C = Capacitor(value=1, unit='pF')
        J = Junction(value=1, unit='GHz', cap=C)
        L = Inductor(value=1, unit='nH')
        c = Circuit([(0, 1, J), (0, 1, L)])
        self.assertTrue(np.allclose(c.F @ c.K, 0))

    def test_coupled_oscillators_frequencies(self):
        """Coupled LC oscillators: exact frequency formula from capacitive coupling."""
        C1  = Capacitor(value=1, unit='pF')
        C2  = Capacitor(value=1, unit='pF')
        Cg  = Capacitor(value=2, unit='pF')
        L1  = Inductor(value=1, unit='nH')
        L2  = Inductor(value=1, unit='nH')
        c   = Circuit([(0, 1, L1), (1, 2, Cg), (2, 0, L2), (0, 1, C1), (2, 0, C2)])
        omega1 = 10 * np.sqrt(2)
        omega2 = 10 * np.sqrt(10)
        H = c.extended_quantum_hamiltonian
        self.assertTrue(np.allclose(
            H, np.diag([omega1, omega2, omega1, omega2]), atol=1e-6))


# ── Linear topology catalogue ─────────────────────────────────────────────────

class TestLinearTopologies(unittest.TestCase):
    """Exact Hamiltonian values for small linear circuits with analytical solutions."""

    def test_triangle_hamiltonian_values(self):
        """Triangle (C on 0-2, L on 0-1 and 1-2) with C=L=1 GHz → H = [[1,0],[0,2]]."""
        C  = Capacitor(value=1, unit='GHz')
        L  = Inductor(value=1, unit='GHz')
        cr = Circuit([(0, 2, C), (0, 1, L), (1, 2, L)])
        self.assertTrue(np.allclose(cr.quadratic_hamiltonian,
                                    np.array([[1., 0.], [0., 2.]])))

    def test_two_caps_one_inductor_parallel(self):
        """2 caps in parallel + 1 inductor: ω = 1/√(2·C·L)."""
        C  = Capacitor(value=1, unit='pF')
        L  = Inductor(value=1, unit='nH')
        cr = Circuit([(0, 1, C), (0, 1, C), (0, 1, L)])
        omega = 1e-9 / np.sqrt(2 * C.cValue * 1e-12 * L.lValue * 1e-9)
        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian,
                                    np.array([[omega, 0], [0, omega]])))

    def test_two_caps_one_inductor_series(self):
        """2 caps in series + 1 inductor: ω = 1/√(C/2 · L) = 1/√(0.5·C·L)."""
        C  = Capacitor(value=1, unit='pF')
        L  = Inductor(value=1, unit='nH')
        cr = Circuit([(0, 1, C), (1, 2, C), (2, 0, L)])
        omega = 1e-9 / np.sqrt(0.5 * C.cValue * 1e-12 * L.lValue * 1e-9)
        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian,
                                    np.array([[omega, 0], [0, omega]])))

    def test_star_circuit(self):
        """Symmetric star (3 caps + 3 inductors) → 2 degenerate modes at ~18.257 GHz."""
        C  = Capacitor(value=1, unit='pF')
        L  = Inductor(value=1, unit='nH')
        cr = Circuit([(0, 1, C), (1, 2, C), (2, 0, C),
                      (0, 3, L), (1, 3, L), (2, 3, L)])
        omega    = 18.2574110
        expected = np.diag([omega, omega, omega, omega])
        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian, expected))


# ── 2. Topology edge cases ────────────────────────────────────────────────────

class TestTopologyEdgeCases(unittest.TestCase):
    """
    Concrete circuits that verify the topology rules about which elements
    create or kill compact modes, and the fixed element ordering invariant.
    """

    def test_no_compact_flux_without_jj(self):
        """Pure LC circuit: no JJ → no compact flux variable."""
        topo = Topology([(0, 1, Inductor(1, 'GHz')), (0, 1, Capacitor(1, 'GHz'))])
        self.assertEqual(topo.no_reduced_compact_flux, 0)

    def test_no_compact_charge_without_qps(self):
        """Pure LC circuit: no QPS → no compact charge variable."""
        topo = Topology([(0, 1, Inductor(1, 'GHz')), (0, 1, Capacitor(1, 'GHz'))])
        self.assertEqual(topo.no_reduced_compact_charge, 0)

    def test_single_jj_one_compact_flux(self):
        """Transmon (one JJ, no parallel inductor): exactly one compact flux."""
        topo = Topology([(0, 1, Junction(1, 'GHz', cap=Capacitor(1, 'GHz')))])
        self.assertEqual(topo.no_reduced_compact_flux, 1)

    def test_single_qps_one_compact_charge(self):
        """Dual-transmon (one QPS): exactly one compact charge."""
        topo = Topology([(0, 1, PhaseSlip(1, 'GHz', L_value=1, L_unit='GHz'))])
        self.assertEqual(topo.no_reduced_compact_charge, 1)

    def test_parallel_inductor_kills_compact_flux(self):
        """Fluxonium (JJ ∥ L): inductor extends the JJ flux → nCF = 0."""
        J = Junction(1, 'GHz', cap=Capacitor(1, 'GHz'))
        topo = Topology([(0, 1, J), (0, 1, Inductor(1, 'GHz'))])
        self.assertEqual(topo.no_reduced_compact_flux, 0)

    def test_parallel_capacitor_kills_compact_charge(self):
        """Dual-fluxonium (QPS ∥ C): capacitor extends the QPS charge → nCC = 0."""
        P = PhaseSlip(1, 'GHz', L_value=1, L_unit='GHz')
        topo = Topology([(0, 1, P), (0, 1, Capacitor(1, 'GHz'))])
        self.assertEqual(topo.no_reduced_compact_charge, 0)

    def test_qps_element_before_its_inductor(self):
        """In a QPS circuit, PhaseSlip (group 2) must appear before Inductor (group 3)."""
        P = PhaseSlip(1, 'GHz', L_value=1, L_unit='GHz')
        topo = Topology([(0, 1, P)])
        self.assertIsInstance(topo.elements[0][2], PhaseSlip)
        self.assertIsInstance(topo.elements[1][2], Inductor)

    def test_jj_element_before_its_capacitor(self):
        """In a JJ circuit, Junction (group 0) must appear before Capacitor (group 1)."""
        J = Junction(1, 'GHz', cap=Capacitor(1, 'GHz'))
        topo = Topology([(0, 1, J)])
        self.assertIsInstance(topo.elements[0][2], Junction)
        self.assertIsInstance(topo.elements[1][2], Capacitor)


# ── 3. Geometry mode counts ───────────────────────────────────────────────────

class TestGeometryModeCounts(unittest.TestCase):
    """Concrete circuits that verify how mode counts emerge from topology."""

    def _geom(self, edges):
        topo = Topology(edges)
        return Geometry(topo)

    def test_lc_one_harmonic_mode(self):
        """LC oscillator → 1 harmonic mode (2 independent variables), 0 compact."""
        geom = self._geom([(0, 1, Inductor(1, 'GHz')), (0, 1, Capacitor(1, 'GHz'))])
        self.assertEqual(geom.no_independent_variables, 2)
        self.assertEqual(geom.no_final_compact_flux,    0)
        self.assertEqual(geom.no_final_compact_charge,  0)

    def test_coupled_lc_two_harmonic_modes(self):
        """Two independent LC loops → 2 harmonic modes (4 independent variables)."""
        geom = self._geom([
            (0, 1, Inductor(1, 'GHz')), (0, 1, Capacitor(1, 'GHz')),
            (0, 2, Inductor(1, 'GHz')), (0, 2, Capacitor(1, 'GHz')),
        ])
        self.assertEqual(geom.no_independent_variables, 4)
        self.assertEqual(geom.no_final_compact_flux,    0)
        self.assertEqual(geom.no_final_compact_charge,  0)

    def test_transmon_one_compact_flux(self):
        """Transmon → 1 compact flux, 0 compact charge, 0 harmonic modes."""
        J = Junction(1, 'GHz', cap=Capacitor(1, 'GHz'))
        geom = self._geom([(0, 1, J)])
        self.assertEqual(geom.no_final_compact_flux,   1)
        self.assertEqual(geom.no_final_compact_charge, 0)

    def test_dual_transmon_one_compact_charge(self):
        """Dual-transmon → 0 compact flux, 1 compact charge, 0 harmonic modes."""
        P = PhaseSlip(1, 'GHz', L_value=1, L_unit='GHz')
        geom = self._geom([(0, 1, P)])
        self.assertEqual(geom.no_final_compact_flux,   0)
        self.assertEqual(geom.no_final_compact_charge, 1)

    def test_fluxonium_one_harmonic_mode(self):
        """Fluxonium (JJ ∥ L) → inductor extends flux → 1 harmonic mode, 0 compact."""
        J = Junction(1, 'GHz', cap=Capacitor(1, 'GHz'))
        geom = self._geom([(0, 1, J), (0, 1, Inductor(1, 'GHz'))])
        self.assertEqual(geom.no_final_compact_flux,    0)
        self.assertEqual(geom.no_final_compact_charge,  0)
        self.assertEqual(geom.no_independent_variables, 2)

    def test_dual_fluxonium_one_harmonic_mode(self):
        """Dual-fluxonium (QPS ∥ C) → capacitor extends charge → 1 harmonic mode, 0 compact."""
        P = PhaseSlip(1, 'GHz', L_value=1, L_unit='GHz')
        geom = self._geom([(0, 1, P), (0, 1, Capacitor(1, 'GHz'))])
        self.assertEqual(geom.no_final_compact_charge,  0)
        self.assertEqual(geom.no_final_compact_flux,    0)
        self.assertEqual(geom.no_independent_variables, 2)

    def test_two_jj_series_two_compact_flux(self):
        """Two JJ in series → 2 independent compact flux modes."""
        J1 = Junction(1, 'GHz', cap=Capacitor(1, 'GHz'))
        J2 = Junction(1, 'GHz', cap=Capacitor(1, 'GHz'))
        geom = self._geom([(0, 1, J1), (1, 2, J2), (0, 2, Capacitor(1, 'GHz'))])
        self.assertEqual(geom.no_final_compact_flux,   2)
        self.assertEqual(geom.no_final_compact_charge, 0)


# ── 4. QPS scaling laws ───────────────────────────────────────────────────────

class TestParallelQPSScaling(unittest.TestCase):
    """Physical scaling laws for N identical QPS elements in parallel."""

    def _build(self, n_qps, el=1.0, ep=0.5):
        qps   = [PhaseSlip(ep, 'GHz', L_value=el, L_unit='GHz') for _ in range(n_qps)]
        topo  = Topology([(0, 1, qps[k]) for k in range(n_qps)])
        geom  = Geometry(topo)
        quant = Quantization(topo, geom)
        return topo, geom, quant

    def test_always_one_compact_charge_mode(self):
        """N parallel QPS on the same node pair → always exactly 1 compact charge mode."""
        for n in (1, 2, 3, 4):
            topo, _, _ = self._build(n)
            self.assertEqual(topo.no_reduced_compact_charge, 1,
                             msg=f"N={n} QPS parallel: expected nCC=1")

    def test_inductive_energy_scales_with_n(self):
        """H[0,0] ∝ N: N parallel inductors raise the inductive energy by N."""
        _, _, q1 = self._build(1, el=1.0)
        _, _, q2 = self._build(2, el=1.0)
        _, _, q3 = self._build(3, el=1.0)
        h1 = q1.quadratic_hamiltonian[0, 0]
        self.assertAlmostEqual(q2.quadratic_hamiltonian[0, 0], 2 * h1, places=10)
        self.assertAlmostEqual(q3.quadratic_hamiltonian[0, 0], 3 * h1, places=10)

    def test_identical_parallel_qps_share_equal_vectors(self):
        """N identical parallel QPS must all have the same coupling vector."""
        for n in (2, 3):
            _, _, quant = self._build(n)
            v = quant.vector_QPS
            for k in range(1, n):
                self.assertTrue(np.allclose(v[:, 0], v[:, k]),
                                msg=f"N={n}: QPS #{k} vector differs from QPS #0")

    def test_qps_groups_one_group_per_node_pair(self):
        """All N QPS on the same node pair must form a single QPS group."""
        topo1, _, _ = self._build(1)
        topo2, _, _ = self._build(2)
        self.assertEqual(len(topo1.qps_groups), 1)
        self.assertEqual(len(topo2.qps_groups), 1)
        self.assertEqual(len(list(topo2.qps_groups.values())[0]), 2)


# ── 5. QPS regression tests ───────────────────────────────────────────────────

def _J():
    return Junction(value=1, unit='GHz', cap=Capacitor(value=1, unit='GHz'))

def _P():
    return PhaseSlip(value=1, unit='GHz', L_value=1, L_unit='GHz')


class TestQPSRegressions(unittest.TestCase):
    """
    Exact nCF / nCC / vector values produced by each of the five QPS bug fixes.

    Bug A — JJ ∥ QPS same nodes:   nCF=0, nCC=1
    Bug B — JJ-QPS chain:           nCF=1, nCC=1
    Bug C — 2JJ + QPS shared node:  vector_QPS non-zero, nCC=1
    Bug D — Ring topologies:         vector non-zero, correct compact counts
    """

    def test_jj_qps_same_nodes_nCF_zero(self):
        """Bug A: QPS inductor extends the JJ flux → nCF = 0."""
        c = Circuit([(0, 1, _J()), (0, 1, _P())])
        self.assertEqual(c.topo.no_reduced_compact_flux, 0)

    def test_jj_qps_same_nodes_nCC_one(self):
        """Bug A: nCC = 1."""
        c = Circuit([(0, 1, _J()), (0, 1, _P())])
        self.assertEqual(c.topo.no_reduced_compact_charge, 1)

    def test_jj_qps_chain_nCF_one(self):
        """Bug B: nCF = 1."""
        c = Circuit([(0, 1, _J()), (1, 2, _P())])
        self.assertEqual(c.topo.no_reduced_compact_flux, 1)

    def test_jj_qps_chain_nCC_one(self):
        """Bug B: nCC = 1."""
        c = Circuit([(0, 1, _J()), (1, 2, _P())])
        self.assertEqual(c.topo.no_reduced_compact_charge, 1)

    def test_2jj_qps_shared_node_nCC_one(self):
        """Bug C: Block-3 null-space fix — QPS charge lands in the dynamical sector."""
        c = Circuit([(0, 1, _J()), (1, 2, _J()), (0, 1, _P())])
        self.assertFalse(np.allclose(c.vector_QPS, 0))
        self.assertEqual(c.topo.no_reduced_compact_charge, 1)

    def test_jj_jj_qps_ring_nCC_one(self):
        """Bug D: JJ-JJ-QPS ring — vector_QPS non-zero, nCC = 1."""
        c = Circuit([(0, 1, _J()), (1, 2, _J()), (2, 0, _P())])
        self.assertFalse(np.allclose(c.vector_QPS, 0))
        self.assertEqual(c.topo.no_reduced_compact_charge, 1)

    def test_qps_qps_jj_ring_compact_counts(self):
        """Bug D: QPS-QPS-JJ ring — two QPS inductors extend JJ flux → nCF=0, nCC=2."""
        c = Circuit([(0, 1, _P()), (1, 2, _P()), (2, 0, _J())])
        self.assertFalse(np.allclose(c.vector_JJ, 0))
        self.assertEqual(c.topo.no_reduced_compact_flux,   0)
        self.assertEqual(c.topo.no_reduced_compact_charge, 2)


if __name__ == '__main__':
    unittest.main()
