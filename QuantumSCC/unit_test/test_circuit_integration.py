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

from QuantumSCC import Circuit, Capacitor, Inductor, Junction


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


if __name__ == '__main__':
    unittest.main()
