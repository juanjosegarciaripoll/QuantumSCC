"""
Integration tests for the Circuit class in QuantumSCC.circuit

Tests the full pipeline (Topology -> Geometry -> Quantization) through the
public Circuit API, verifying that attribute delegation, physical results,
and output methods are all consistent.
"""

import unittest
import numpy as np
import os
import sys
from io import StringIO
import contextlib

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip
from QuantumSCC.utils import units as unt


class TestCircuitLC(unittest.TestCase):
    """LC oscillator — simplest complete circuit (1 mode)."""

    def setUp(self):
        self.C = Capacitor(value=1, unit='pF')
        self.L = Inductor(value=1, unit='nH')
        self.circuit = Circuit([(0, 1, self.L), (0, 1, self.C)])

    # --- structure ---

    def test_attribute_delegation_topology(self):
        """Topological attributes exposed on Circuit are the same objects as in Topology."""
        c = self.circuit
        self.assertIs(c.Fcut, c.topo.Fcut)
        self.assertIs(c.Floop, c.topo.Floop)
        self.assertIs(c.F, c.topo.F)
        self.assertIs(c.K, c.topo.K)

    def test_attribute_delegation_geometry(self):
        """Geometric attributes exposed on Circuit are the same objects as in Geometry."""
        c = self.circuit
        self.assertIs(c.omega_2B,        c.geom.omega_2B)
        self.assertIs(c.omega_symplectic, c.geom.omega_symplectic)
        self.assertIs(c.V,               c.geom.V)

    def test_attribute_delegation_quantization(self):
        """Quantization attributes exposed on Circuit are the same objects as in Quantization."""
        c = self.circuit
        self.assertIs(c.quadratic_hamiltonian,      c.quant.quadratic_hamiltonian)
        self.assertIs(c.extended_quantum_hamiltonian, c.quant.extended_quantum_hamiltonian)
        self.assertIs(c.vector_JJ,                  c.quant.vector_JJ)

    # --- topology ---

    def test_element_counts(self):
        self.assertEqual(self.circuit.no_JJ,         0)
        self.assertEqual(self.circuit.no_Capacitors,  1)
        self.assertEqual(self.circuit.no_Inductors,   1)
        self.assertEqual(self.circuit.no_elements,    2)

    def test_kirchhoff_constraint(self):
        """Full Kirchhoff constraint F @ K must be zero."""
        product = self.circuit.F @ self.circuit.K
        self.assertTrue(np.allclose(product, 0),
                        msg=f"F @ K max error: {np.max(np.abs(product)):.2e}")

    # --- modes ---

    def test_one_dynamic_mode(self):
        """LC oscillator has exactly 1 dynamic mode (2 independent variables)."""
        self.assertEqual(self.circuit.no_independent_variables, 2)

    def test_no_junctions_vector_empty(self):
        """With no junctions, vector_JJ has 0 columns."""
        self.assertEqual(self.circuit.vector_JJ.shape[1], 0)

    # --- frequency ---

    def test_frequency_matches_theory(self):
        """Mode frequency equals omega = 1e-9 / sqrt(L*C) in GHz."""
        val_c = self.C.cValue * 1e-12   # pF -> F
        val_l = self.L.lValue * 1e-9    # nH -> H
        omega_theory = 1e-9 / np.sqrt(val_c * val_l)
        H = self.circuit.extended_quantum_hamiltonian.real
        self.assertTrue(np.allclose(np.diag(H), omega_theory),
                        msg=f"Expected {omega_theory:.4f}, got {np.diag(H)}")

    # --- display ---

    def test_diagonal_hamiltonian_display(self):
        """diagonal_harmonic_Hamiltonian_expression() runs and produces output."""
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            self.circuit.diagonal_harmonic_Hamiltonian_expression()
        self.assertGreater(len(buf.getvalue()), 0)


class TestCircuitTwoLC(unittest.TestCase):
    """Two independent LC oscillators sharing a common ground node (2 modes, no coupling)."""

    def setUp(self):
        C = Capacitor(value=1, unit='pF')
        L = Inductor(value=1, unit='nH')
        # (0,1) and (0,2): two separate loops, no capacitor loop between them
        elements = [
            (0, 1, L), (0, 1, C),
            (0, 2, L), (0, 2, C),
        ]
        self.circuit = Circuit(elements)
        # Reference frequency for both (identical) oscillators
        self.omega = 1e-9 / np.sqrt(1e-12 * 1e-9)

    def test_two_dynamic_modes(self):
        """Two LC loops give 4 independent variables (2 modes)."""
        self.assertEqual(self.circuit.no_independent_variables, 4)

    def test_hamiltonian_shape(self):
        """Quadratic Hamiltonian is 4x4 for a 2-mode circuit."""
        self.assertEqual(self.circuit.quadratic_hamiltonian.shape, (4, 4))

    def test_hamiltonian_symmetric(self):
        H = self.circuit.quadratic_hamiltonian
        self.assertTrue(np.allclose(H, H.T))

    def test_quantum_hamiltonian_shape(self):
        """Extended quantum Hamiltonian is 4x4 (2 mode pairs)."""
        self.assertEqual(self.circuit.extended_quantum_hamiltonian.shape, (4, 4))

    def test_both_frequencies_positive(self):
        """Both mode frequencies must be strictly positive."""
        H = self.circuit.extended_quantum_hamiltonian.real
        n = H.shape[0] // 2
        freqs = [H[i, i] for i in range(n)]
        self.assertTrue(all(f > 0 for f in freqs),
                        msg=f"Negative or zero frequency found: {freqs}")

    def test_kirchhoff_constraint(self):
        product = self.circuit.F @ self.circuit.K
        self.assertTrue(np.allclose(product, 0))


class TestCircuitFluxonium(unittest.TestCase):
    """Fluxonium (J ∥ L) — nonlinear circuit with 1 compact flux variable."""

    def setUp(self):
        C_J = Capacitor(value=1, unit='pF')
        J   = Junction(value=1, unit='GHz', cap=C_J)
        L   = Inductor(value=1, unit='nH')
        self.circuit = Circuit([(0, 1, J), (0, 1, L)])

    def test_element_counts(self):
        self.assertEqual(self.circuit.no_JJ,        1)
        self.assertEqual(self.circuit.no_Capacitors, 1)  # the JJ's parallel cap
        self.assertEqual(self.circuit.no_Inductors,  1)

    def test_jj_vector_nonzero(self):
        """JJ coupling vector must not be zero."""
        self.assertFalse(np.allclose(self.circuit.vector_JJ, 0),
                         msg="vector_JJ is zero — junction decoupled from dynamics")

    def test_jj_vector_single_column(self):
        """One junction -> vector_JJ has exactly 1 column."""
        self.assertEqual(self.circuit.vector_JJ.shape[1], 1)

    def test_hamiltonian_phiq_square(self):
        """FS_quadratic_hamiltonian_phiq must be square."""
        H = self.circuit.FS_quadratic_hamiltonian_phiq
        self.assertEqual(H.shape[0], H.shape[1])

    def test_kirchhoff_constraint(self):
        product = self.circuit.F @ self.circuit.K
        self.assertTrue(np.allclose(product, 0))

    def test_hamiltonian_expression_prints_cos(self):
        """Hamiltonian_expression() must print the cos(...) Josephson term."""
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            self.circuit.Hamiltonian_expression()
        self.assertIn('cos', buf.getvalue(),
                      msg="Josephson cos term not found in Hamiltonian_expression output")


class TestCircuitDualLC(unittest.TestCase):
    """
    Dual-transmon: QPS(1 GHz) ∥ Inductor(1 nH).
    H/ℏ = E_L φ² − E_P cos(q/e).
    H[0,0] = 2·E_L_code, H[1,1] = 0 (compact charge).
    """

    def setUp(self):
        self.L_nH = 1.0
        L = Inductor(value=self.L_nH, unit='nH')
        P = PhaseSlip(value=1.0, unit='GHz', ind=L)
        self.circuit  = Circuit([(0, 1, P)])
        self.E_L_code = (unt.Phi0 / (2 * np.pi))**2 / (
            2 * self.L_nH * 1e-9 * unt.hbar) / 1e9

    def test_element_counts(self):
        self.assertEqual(self.circuit.no_QPS,        1)
        self.assertEqual(self.circuit.no_JJ,         0)
        self.assertEqual(self.circuit.no_Inductors,   1)
        self.assertEqual(self.circuit.no_Capacitors,  0)

    def test_kirchhoff_constraint(self):
        product = self.circuit.F @ self.circuit.K
        self.assertTrue(np.allclose(product, 0),
                        msg=f"F@K max: {np.max(np.abs(product)):.2e}")

    def test_one_dynamic_mode(self):
        self.assertEqual(self.circuit.no_independent_variables, 2)

    def test_compact_charge_count(self):
        self.assertEqual(self.circuit.no_final_compact_charge, 1)

    def test_compact_flux_count(self):
        self.assertEqual(self.circuit.no_final_compact_flux, 0)

    def test_flux_entry_matches_formula(self):
        """H[0,0] = 2·E_L_code."""
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


class TestQPSJJDuality(unittest.TestCase):
    """
    Structural duality: JJ (compact flux) ↔ QPS (compact charge).
      Transmon:      H[0,0]=0, H[1,1]=4·E_C_code (compact flux)
      Dual-transmon: H[1,1]=0, H[0,0]=2·E_L_code (compact charge)
    """

    def setUp(self):
        C = Capacitor(value=1, unit='pF')
        J = Junction(value=1, unit='GHz', cap=C)
        self.transmon = Circuit([(0, 1, J)])
        self.E_C_code = (2 * unt.e)**2 / (2 * 1e-12 * unt.hbar) / 1e9

        L = Inductor(value=1, unit='nH')
        P = PhaseSlip(value=1, unit='GHz', ind=L)
        self.dual = Circuit([(0, 1, P)])
        self.E_L_code = (unt.Phi0 / (2 * np.pi))**2 / (2 * 1e-9 * unt.hbar) / 1e9

    def test_transmon_compact_flux_zero(self):
        self.assertAlmostEqual(abs(self.transmon.quadratic_hamiltonian[0, 0]), 0.0, delta=1e-10)

    def test_transmon_charge_entry_formula(self):
        """H[1,1] = 4·E_C_code (GS normalisation + V transform)."""
        H = self.transmon.quadratic_hamiltonian
        self.assertAlmostEqual(H[1, 1].real, 4.0 * self.E_C_code, delta=self.E_C_code * 1e-5)

    def test_dual_compact_charge_zero(self):
        self.assertAlmostEqual(abs(self.dual.quadratic_hamiltonian[1, 1]), 0.0, delta=1e-10)

    def test_dual_flux_entry_formula(self):
        """H[0,0] = 2·E_L_code."""
        H = self.dual.quadratic_hamiltonian
        self.assertAlmostEqual(H[0, 0].real, 2.0 * self.E_L_code, delta=self.E_L_code * 1e-5)

    def test_compact_variable_counts(self):
        self.assertEqual(self.transmon.no_final_compact_flux,   1)
        self.assertEqual(self.transmon.no_final_compact_charge,  0)
        self.assertEqual(self.dual.no_final_compact_flux,        0)
        self.assertEqual(self.dual.no_final_compact_charge,      1)

    def test_kirchhoff_both(self):
        self.assertTrue(np.allclose(self.transmon.F @ self.transmon.K, 0))
        self.assertTrue(np.allclose(self.dual.F    @ self.dual.K,    0))


class TestCircuitQPSErrors(unittest.TestCase):
    """Unsupported QPS topologies must raise clear errors."""

    def test_jj_and_qps_in_parallel_builds(self):
        """JJ ∥ QPS on same nodes: now supported — circuit must build without error."""
        C = Capacitor(value=1, unit='pF')
        J = Junction(value=1, unit='GHz', cap=C)
        L = Inductor(value=1, unit='nH')
        P = PhaseSlip(value=1, unit='GHz', ind=L)
        circuit = Circuit([(0, 1, J), (0, 1, P)])
        self.assertIsNotNone(circuit.quadratic_hamiltonian)

    def test_phaseslip_no_inductor_raises(self):
        with self.assertRaises(ValueError):
            PhaseSlip(value=1, unit='GHz')

    def test_phaseslip_bad_unit_raises(self):
        with self.assertRaises(ValueError):
            PhaseSlip(value=1, unit='nH', ind=Inductor(1, unit='nH'))


class TestQPSBackwardsCompatibility(unittest.TestCase):
    """Existing circuits must be unaffected by the QPS extension."""

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
        C1 = Capacitor(value=1, unit='pF')
        C2 = Capacitor(value=1, unit='pF')
        Cg = Capacitor(value=2, unit='pF')
        L1 = Inductor(value=1, unit='nH')
        L2 = Inductor(value=1, unit='nH')
        c = Circuit([(0, 1, L1), (1, 2, Cg), (2, 0, L2), (0, 1, C1), (2, 0, C2)])
        omega1 = 10 * np.sqrt(2)
        omega2 = 10 * np.sqrt(10)
        H = c.extended_quantum_hamiltonian
        self.assertTrue(np.allclose(
            H, np.diag([omega1, omega2, omega1, omega2]), atol=1e-6))


class TestCircuitLinearTopologies(unittest.TestCase):
    """Additional linear circuit topologies — migrated from legacy tests/test_circuit.py."""

    def test_triangle_hamiltonian_values(self):
        """Triangle: C(0-2), L(0-1), L(1-2) with C=L=1 GHz → H = [[1,0],[0,2]]."""
        C = Capacitor(value=1, unit='GHz')
        L = Inductor(value=1, unit='GHz')
        cr = Circuit([(0, 2, C), (0, 1, L), (1, 2, L)])
        expected = np.array([[1., 0.], [0., 2.]])
        self.assertTrue(np.allclose(cr.quadratic_hamiltonian, expected))

    def test_2C_and1L_parallel(self):
        """2 caps in parallel + 1 inductor: omega = 1/sqrt(2*C*L)."""
        C = Capacitor(value=1, unit='pF')
        L = Inductor(value=1, unit='nH')
        cr = Circuit([(0, 1, C), (0, 1, C), (0, 1, L)])
        omega = 1e-9 / np.sqrt(2 * C.cValue * 1e-12 * L.lValue * 1e-9)
        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian,
                                    np.array([[omega, 0], [0, omega]])))

    def test_2C_and1L_series(self):
        """2 caps in series + 1 inductor: omega = 1/sqrt(0.5*C*L)."""
        C = Capacitor(value=1, unit='pF')
        L = Inductor(value=1, unit='nH')
        cr = Circuit([(0, 1, C), (1, 2, C), (2, 0, L)])
        omega = 1e-9 / np.sqrt(0.5 * C.cValue * 1e-12 * L.lValue * 1e-9)
        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian,
                                    np.array([[omega, 0], [0, omega]])))

    def test_star_circuit(self):
        """Symmetric star: 3 caps + 3 inductors → 2 degenerate modes at ~18.257 GHz."""
        C = Capacitor(value=1, unit='pF')
        L = Inductor(value=1, unit='nH')
        cr = Circuit([(0, 1, C), (1, 2, C), (2, 0, C), (0, 3, L), (1, 3, L), (2, 3, L)])
        omega = 18.2574110
        expected = np.diag([omega, omega, omega, omega])
        self.assertTrue(np.allclose(cr.extended_quantum_hamiltonian, expected))


if __name__ == '__main__':
    unittest.main()
