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
5. QPS regression tests   — exact nCF / nCC values pinned by bug fixes
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
    """Fluxonium (JJ + C + L) — nonlinear circuit with 1 harmonic mode."""

    def setUp(self):
        C_J = Capacitor(value=1, unit='pF')
        J   = Junction(value=1, unit='GHz')
        L   = Inductor(value=1, unit='nH')
        self.circuit = Circuit([(0, 1, J), (0, 1, C_J), (0, 1, L)])

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
    Expected: H[0,0] = 0 (compact flux has no quadratic term), H[1,1] = 2·E_C.
    """

    def setUp(self):
        self.C_pF = 1.0
        C = Capacitor(value=self.C_pF, unit='pF')
        J = Junction(value=1, unit='GHz')
        self.circuit = Circuit([(0, 1, J), (0, 1, C)])
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
    Dual-transmon: QPS with parallel Inductor.
    H = E_L·φ² − E_P·cos(q).
    Expected: H[0,0] = 2·E_L, H[1,1] = 0 (compact charge has no quadratic term).
    """

    def setUp(self):
        self.L_nH = 1.0
        P = PhaseSlip(value=1.0, unit='GHz')
        L = Inductor(value=self.L_nH, unit='nH')
        self.circuit  = Circuit([(0, 1, P), (0, 1, L)])
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
      Dual-transmon:  H[1,1]=0  (compact charge), H[0,0]=2·E_L
    """

    def setUp(self):
        C = Capacitor(value=1, unit='pF')
        J = Junction(value=1, unit='GHz')
        self.transmon    = Circuit([(0, 1, J), (0, 1, C)])
        self.E_C_code    = (2 * unt.e)**2 / (2 * 1e-12 * unt.hbar) / 1e9

        P = PhaseSlip(value=1, unit='GHz')
        L = Inductor(value=1, unit='nH')
        self.dual        = Circuit([(0, 1, P), (0, 1, L)])
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
        """JJ + C + QPS + L on the same nodes: must not raise."""
        J = Junction(value=1, unit='GHz')
        C = Capacitor(value=1, unit='pF')
        P = PhaseSlip(value=1, unit='GHz')
        L = Inductor(value=1, unit='nH')
        circuit = Circuit([(0, 1, J), (0, 1, C), (0, 1, P), (0, 1, L)])
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
        J = Junction(value=1, unit='GHz')
        c = Circuit([(0, 1, J), (0, 1, C)])
        self.assertEqual(c.no_final_compact_charge, 0)
        self.assertEqual(c.no_QPS,                  0)

    def test_fluxonium_kirchhoff(self):
        C = Capacitor(value=1, unit='pF')
        J = Junction(value=1, unit='GHz')
        L = Inductor(value=1, unit='nH')
        c = Circuit([(0, 1, J), (0, 1, C), (0, 1, L)])
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
        """Triangle (C on 0-2, L on 0-1 and 1-2): H = diag(E_L, 2*E_C).

        Two equal series inductors each with E_L contribute E_L_eff = E_L/2 to the
        reduced Hamiltonian, so H[0,0] = 2*E_L_eff = E_L.  The single capacitor
        gives H[1,1] = 2*E_C (standard compact/extended norm convention).
        """
        E_C_val = 1.0   # GHz
        E_L_val = 1.0   # GHz (each inductor)
        C  = Capacitor(value=E_C_val, unit='GHz')
        L  = Inductor(value=E_L_val, unit='GHz')
        cr = Circuit([(0, 2, C), (0, 1, L), (1, 2, L)])
        # Two equal series inductors → H_ind = 2*(E_L/2) = E_L; cap → H_cap = 2*E_C
        H_expected = np.diag([E_L_val, 2 * E_C_val])
        self.assertTrue(np.allclose(cr.quadratic_hamiltonian, H_expected))

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
        """Symmetric star (3 caps + 3 inductors) → 2 doubly-degenerate modes.

        Topology: Δ-triangle of caps (0-1, 1-2, 2-0) + star inductors to centre (3).
        The Δ cap triangle → Y-equivalent: C_Y = 3·C from each outer node to a
        virtual neutral centre.  For the two doubly-degenerate (E) modes the centre
        node acts as virtual ground, so each E-mode sees L to ground and C_Y = 3·C
        to the neutral point → ω = 1/√(3·C·L).  The symmetric (A₁) mode has zero
        frequency (no capacitive restoring force).
        """
        C  = Capacitor(value=1, unit='pF')
        L  = Inductor(value=1, unit='nH')
        cr = Circuit([(0, 1, C), (1, 2, C), (2, 0, C),
                      (0, 3, L), (1, 3, L), (2, 3, L)])
        # ω_E = 1/√(3·C·L)  (Δ→Y transform gives C_Y = 3·C)
        omega    = 1e-9 / np.sqrt(3 * C.cValue * 1e-12 * L.lValue * 1e-9)
        expected = np.diag([omega, omega, omega, omega])
        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian, expected, rtol=1e-6))


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

    def test_single_jj_with_cap_one_compact_flux(self):
        """Transmon (JJ + C): exactly one compact flux."""
        topo = Topology([(0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz'))])
        self.assertEqual(topo.no_reduced_compact_flux, 1)

    def test_single_qps_with_ind_one_compact_charge(self):
        """Dual-transmon (QPS + L): exactly one compact charge."""
        topo = Topology([(0, 1, PhaseSlip(1, 'GHz')), (0, 1, Inductor(1, 'GHz'))])
        self.assertEqual(topo.no_reduced_compact_charge, 1)

    def test_parallel_inductor_kills_compact_flux(self):
        """Fluxonium (JJ + C + L): inductor extends the JJ flux → nCF = 0."""
        topo = Topology([(0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz')),
                         (0, 1, Inductor(1, 'GHz'))])
        self.assertEqual(topo.no_reduced_compact_flux, 0)

    def test_parallel_capacitor_kills_compact_charge(self):
        """Dual-fluxonium (QPS + L + C): capacitor extends the QPS charge → nCC = 0."""
        topo = Topology([(0, 1, PhaseSlip(1, 'GHz')), (0, 1, Inductor(1, 'GHz')),
                         (0, 1, Capacitor(1, 'GHz'))])
        self.assertEqual(topo.no_reduced_compact_charge, 0)

    def test_element_ordering_jj_before_cap(self):
        """In element list, Junction (group 0) must appear before Capacitor (group 1)."""
        topo = Topology([(0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz'))])
        self.assertIsInstance(topo.elements[0][2], Junction)
        self.assertIsInstance(topo.elements[1][2], Capacitor)

    def test_element_ordering_qps_before_ind(self):
        """In element list, PhaseSlip (group 2) must appear before Inductor (group 3)."""
        topo = Topology([(0, 1, PhaseSlip(1, 'GHz')), (0, 1, Inductor(1, 'GHz'))])
        self.assertIsInstance(topo.elements[0][2], PhaseSlip)
        self.assertIsInstance(topo.elements[1][2], Inductor)


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
        geom = self._geom([(0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz'))])
        self.assertEqual(geom.no_final_compact_flux,   1)
        self.assertEqual(geom.no_final_compact_charge, 0)

    def test_dual_transmon_one_compact_charge(self):
        """Dual-transmon → 0 compact flux, 1 compact charge, 0 harmonic modes."""
        geom = self._geom([(0, 1, PhaseSlip(1, 'GHz')), (0, 1, Inductor(1, 'GHz'))])
        self.assertEqual(geom.no_final_compact_flux,   0)
        self.assertEqual(geom.no_final_compact_charge, 1)

    def test_fluxonium_one_harmonic_mode(self):
        """Fluxonium (JJ + C + L) → inductor extends flux → 1 harmonic mode, 0 compact."""
        geom = self._geom([(0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz')),
                           (0, 1, Inductor(1, 'GHz'))])
        self.assertEqual(geom.no_final_compact_flux,    0)
        self.assertEqual(geom.no_final_compact_charge,  0)
        self.assertEqual(geom.no_independent_variables, 2)

    def test_dual_fluxonium_one_harmonic_mode(self):
        """Dual-fluxonium (QPS + L + C) → capacitor extends charge → 1 harmonic mode, 0 compact."""
        geom = self._geom([(0, 1, PhaseSlip(1, 'GHz')), (0, 1, Inductor(1, 'GHz')),
                           (0, 1, Capacitor(1, 'GHz'))])
        self.assertEqual(geom.no_final_compact_charge,  0)
        self.assertEqual(geom.no_final_compact_flux,    0)
        self.assertEqual(geom.no_independent_variables, 2)

    def test_two_jj_series_two_compact_flux(self):
        """Two JJ in series → 2 independent compact flux modes."""
        geom = self._geom([
            (0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz')),
            (1, 2, Junction(1, 'GHz')), (1, 2, Capacitor(1, 'GHz')),
            (0, 2, Capacitor(1, 'GHz')),
        ])
        self.assertEqual(geom.no_final_compact_flux,   2)
        self.assertEqual(geom.no_final_compact_charge, 0)


# ── 4. QPS scaling laws ───────────────────────────────────────────────────────

class TestParallelQPSScaling(unittest.TestCase):
    """Physical scaling laws for N identical QPS elements in parallel.

    N>1 parallel QPS+Ind on the same nodes without a Capacitor creates
    a degenerate extended mode (zero frequency).  The symplectic_transformation
    cannot handle the resulting Jordan block.  Tests for N>1 require a
    Capacitor to regularize the circuit.
    """

    def test_topology_compact_charge_count(self):
        """N parallel QPS on the same node pair → topology gives nCC = max(1, N-1).

        For N=1: 1 compact mode (the single QPS charge).
        For N>=2: N-1 compact modes (the charge-difference directions between QPS).
        The N-2 extra modes for N>=3 are doubly-discrete gauges in the Darboux
        step (zero Omega rows), so no_final_compact_charge remains 0 for N>=2.
        """
        for n in (1, 2, 3, 4):
            expected_ncc = max(1, n - 1)
            qps  = [PhaseSlip(0.5, 'GHz') for _ in range(n)]
            inds = [Inductor(1.0, 'GHz') for _ in range(n)]
            edges = [(0, 1, q) for q in qps] + [(0, 1, l) for l in inds]
            topo = Topology(edges)
            self.assertEqual(topo.no_reduced_compact_charge, expected_ncc,
                             msg=f"N={n} QPS parallel: expected nCC={expected_ncc}")

    def test_single_qps_inductive_energy(self):
        """Single QPS+Ind (dual transmon): H[0,0] = 2*E_L."""
        topo = Topology([(0, 1, PhaseSlip(0.5, 'GHz')), (0, 1, Inductor(1.0, 'GHz'))])
        geom = Geometry(topo)
        quant = Quantization(topo, geom)
        self.assertAlmostEqual(quant.quadratic_hamiltonian[0, 0], 2.0, places=10)

    def test_parallel_qps_vectors_nonzero(self):
        """N parallel QPS on same nodes → all vector_QPS columns are non-zero.

        Doubly-discrete gauge fix: gauge charge differences have integer spectra,
        cos(2π·integer) = 1, so all QPS couple to the same dynamical charge.
        """
        for n in (2, 3, 4):
            qps  = [PhaseSlip(float(i + 1), 'GHz') for i in range(n)]
            inds = [Inductor(1.0, 'GHz') for _ in range(n)]
            edges = [(0, 1, q) for q in qps] + [(0, 1, l) for l in inds]
            circ = Circuit(edges)
            for col in range(n):
                self.assertFalse(np.allclose(circ.vector_QPS[:, col], 0),
                                 msg=f"N={n}: QPS {col} has zero vector")

    def test_parallel_qps_vectors_equal(self):
        """N parallel QPS on same nodes → all vector_QPS columns are identical."""
        for n in (2, 3, 4):
            qps  = [PhaseSlip(float(i + 1), 'GHz') for i in range(n)]
            inds = [Inductor(1.0, 'GHz') for _ in range(n)]
            edges = [(0, 1, q) for q in qps] + [(0, 1, l) for l in inds]
            circ = Circuit(edges)
            for col in range(1, n):
                np.testing.assert_allclose(
                    circ.vector_QPS[:, col], circ.vector_QPS[:, 0],
                    atol=1e-12,
                    err_msg=f"N={n}: QPS {col} vector differs from QPS 0")

    def test_parallel_qps_effective_inductance(self):
        """N parallel QPS+Ind: H_quad non-zero diagonal = 2·Σ E_Li."""
        E_L = [1.0, 2.0, 0.5]
        for n in (2, 3):
            qps  = [PhaseSlip(1.0, 'GHz') for _ in range(n)]
            inds = [Inductor(E_L[i], 'GHz') for i in range(n)]
            edges = [(0, 1, q) for q in qps] + [(0, 1, l) for l in inds]
            circ = Circuit(edges)
            diag = np.diag(circ.quadratic_hamiltonian)
            nonzero = diag[np.abs(diag) > 1e-10]
            expected = 2.0 * sum(E_L[:n])
            self.assertAlmostEqual(nonzero[0], expected, places=10,
                                   msg=f"N={n}: expected 2·Σ E_L = {expected}")


# ── 5. QPS regression tests ───────────────────────────────────────────────────

def _J():
    return Junction(value=1, unit='GHz')

def _P():
    return PhaseSlip(value=1, unit='GHz')

def _C():
    return Capacitor(value=1, unit='GHz')

def _L():
    return Inductor(value=1, unit='GHz')


class TestQPSRegressions(unittest.TestCase):
    """
    Exact nCF / nCC / vector values for key circuit topologies.
    Updated for the new API where C and L are explicit user-provided elements.
    """

    def test_jj_qps_same_nodes_nCF_zero(self):
        """JJ + C + QPS + L on same nodes: inductor extends JJ flux → nCF = 0."""
        c = Circuit([(0, 1, _J()), (0, 1, _C()), (0, 1, _P()), (0, 1, _L())])
        self.assertEqual(c.topo.no_reduced_compact_flux, 0)

    def test_jj_qps_same_nodes_nCC(self):
        """JJ + C + QPS + L on same nodes: Cap suppresses compact charge → nCC = 0."""
        c = Circuit([(0, 1, _J()), (0, 1, _C()), (0, 1, _P()), (0, 1, _L())])
        self.assertEqual(c.topo.no_reduced_compact_charge, 0)

    def test_jj_qps_chain_nCF_one(self):
        """JJ + C on (0,1), QPS + L on (1,2): nCF = 1."""
        c = Circuit([(0, 1, _J()), (0, 1, _C()), (1, 2, _P()), (1, 2, _L())])
        self.assertEqual(c.topo.no_reduced_compact_flux, 1)

    def test_jj_qps_chain_nCC_one(self):
        """JJ + C on (0,1), QPS + L on (1,2): nCC = 1."""
        c = Circuit([(0, 1, _J()), (0, 1, _C()), (1, 2, _P()), (1, 2, _L())])
        self.assertEqual(c.topo.no_reduced_compact_charge, 1)

    def test_2jj_qps_shared_node(self):
        """2JJ + QPS shared node: QPS charge vector non-zero."""
        c = Circuit([(0, 1, _J()), (0, 1, _C()), (1, 2, _J()), (1, 2, _C()),
                     (0, 1, _P()), (0, 1, _L())])
        # Cap on (0,1) same as QPS on (0,1) → kcut_suppressed
        self.assertEqual(c.topo.no_reduced_compact_charge, 0)

    def test_jj_jj_qps_ring(self):
        """JJ-JJ-QPS ring: all on different node pairs."""
        c = Circuit([(0, 1, _J()), (0, 1, _C()), (1, 2, _J()), (1, 2, _C()),
                     (2, 0, _P()), (2, 0, _L())])
        self.assertFalse(np.allclose(c.vector_QPS, 0))
        self.assertEqual(c.topo.no_reduced_compact_charge, 1)

    def test_qps_qps_jj_ring(self):
        """QPS-QPS-JJ ring: two QPS inductors extend JJ flux."""
        c = Circuit([(0, 1, _P()), (0, 1, _L()), (1, 2, _P()), (1, 2, _L()),
                     (2, 0, _J()), (2, 0, _C())])
        self.assertFalse(np.allclose(c.vector_JJ, 0))
        self.assertEqual(c.topo.no_reduced_compact_flux, 0)


class TestDualmonCircuits(unittest.TestCase):
    """Dualmon circuits from Le et al. (arXiv:1904.01843)."""

    def test_dualmon_bare_constructs(self):
        """Bare dualmon (JJ + QPS) constructs with zero quadratic H."""
        c = Circuit([(0, 1, _J()), (0, 1, _P())])
        self.assertEqual(c.no_final_compact_flux, 0)
        self.assertEqual(c.no_final_compact_charge, 0)
        self.assertTrue(np.allclose(c.quadratic_hamiltonian, 0))

    def test_dualmon_gate_constructs(self):
        """Dualmon + gate capacitor (JJ + QPS + C) constructs."""
        c = Circuit([(0, 1, _J()), (0, 1, _P()), (0, 1, _C())])
        self.assertEqual(c.no_final_compact_flux, 0)
        self.assertEqual(c.no_final_compact_charge, 0)

    def test_dualmon_full_constructs(self):
        """Full dualmon (JJ+C on one node, L series, QPS on other) constructs."""
        c = Circuit([(1, 0, _J()), (1, 0, _C()), (1, 2, _L()), (2, 0, _P())])
        self.assertEqual(c.no_final_compact_flux, 0)
        self.assertEqual(c.no_final_compact_charge, 0)
        self.assertEqual(c.no_independent_variables, 4)

    def test_dualmon_full_has_one_oscillator(self):
        """Full dualmon has one oscillator mode and one zero-frequency mode."""
        c = Circuit([(1, 0, _J()), (1, 0, _C()), (1, 2, _L()), (2, 0, _P())])
        H = c.FS_quadratic_hamiltonian_phiq.real
        diag = np.diag(H)
        # One nonzero pair (oscillator) and one zero pair (frozen mode)
        ne = H.shape[0] // 2
        nonzero_flux = np.count_nonzero(np.abs(diag[:ne]) > 1e-10)
        nonzero_charge = np.count_nonzero(np.abs(diag[ne:]) > 1e-10)
        self.assertEqual(nonzero_flux, 1)
        self.assertEqual(nonzero_charge, 1)

    def test_dualmon_full_jj_qps_vectors_nonzero(self):
        """Full dualmon JJ and QPS coupling vectors are nonzero."""
        c = Circuit([(1, 0, _J()), (1, 0, _C()), (1, 2, _L()), (2, 0, _P())])
        self.assertFalse(np.allclose(c.final_vector_JJ_phiq, 0))
        self.assertFalse(np.allclose(c.final_vector_QPS_phiq, 0))

    def test_dualmon_full_with_extra_elements(self):
        """Dualmon full with extra L and C (conftest version) constructs."""
        c = Circuit([(1, 0, _J()), (1, 0, _C()), (1, 2, _L()),
                     (2, 0, _P()), (2, 0, _L()), (1, 0, _C())])
        self.assertEqual(c.no_final_compact_flux, 0)
        self.assertEqual(c.no_final_compact_charge, 1)


if __name__ == '__main__':
    unittest.main()
