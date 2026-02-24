"""
Unit tests for the Quantization class in QuantumSCC.core.quantization
"""

import unittest
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC.core.elements import Capacitor, Inductor, Junction
from QuantumSCC.core.topology import Topology
from QuantumSCC.core.geometry import Geometry
from QuantumSCC.core.quantization import Quantization

class TestQuantizationLinear(unittest.TestCase):
    """Tests for linear circuits (Harmonic Oscillators)."""

    def setUp(self):
        """Set up an LC Oscillator."""
        # C = 1 pF, L = 1 nH
        self.C = Capacitor(value=1, unit='pF')
        self.L = Inductor(value=1, unit='nH')
        self.elements = [(0, 1, self.L), (0, 1, self.C)]
        
        # Initialize dependencies
        self.topo = Topology(self.elements)
        self.geom = Geometry(self.topo)
        
        # Initialize Quantization (Class under test)
        self.quant = Quantization(self.topo, self.geom)

    def test_classical_hamiltonian_properties(self):
        """Check properties of the classical quadratic Hamiltonian."""
        H = self.quant.quadratic_hamiltonian
        
        # 1. Symmetry: H should be symmetric
        self.assertTrue(np.allclose(H, H.T), "Classical Hamiltonian must be symmetric")
        
        # 2. Dimensions: LC has 1 mode -> 2 variables (phi, q) -> 2x2 Matrix
        self.assertEqual(H.shape, (2, 2))
        
        # 3. Positive Definite (Energy > 0)
        eigenvals = np.linalg.eigvals(H)
        self.assertTrue(np.all(eigenvals > 0), "Hamiltonian eigenvalues should be positive for stable LC")

    def test_lc_frequency_correctness(self):
        """Verify the calculated quantum frequency matches theory."""
        # Theoretical Omega = 1 / sqrt(L*C)
        # Note: units in code are normalized to GHz usually, but let's check the raw calculation logic match
        # Using the formula from the previous test suite:
        # omega_theory_GHz = 1e-9 / sqrt(C_F * L_H)
        
        val_c = self.C.cValue * 1e-12 # pF to F
        val_l = self.L.lValue * 1e-9  # nH to H
        omega_theory = 1e-9 / np.sqrt(val_c * val_l)
        
        # The extended quantum hamiltonian should be diagonal with these values
        H_quant = self.quant.extended_quantum_hamiltonian.real
        
        # Check diagonal elements
        diag_elements = np.diag(H_quant)
        
        self.assertTrue(np.allclose(diag_elements, omega_theory),
                        msg=f"Frequency Mismatch. Got: {diag_elements[0]:.4f}, Expected: {omega_theory:.4f}")


class TestQuantizationNonLinear(unittest.TestCase):
    """Tests for nonlinear circuits (Josephson Junctions)."""

    def setUp(self):
        """Set up a Fluxonium (L || J)."""
        self.C_J = Capacitor(value=1, unit='pF')
        self.J = Junction(value=1, unit='GHz', cap=self.C_J)
        self.L = Inductor(value=1, unit='nH')
        
        # Topology: Inductor parallel to Junction
        self.elements = [(0, 1, self.J), (0, 1, self.L)]
        
        self.topo = Topology(self.elements)
        self.geom = Geometry(self.topo)
        self.quant = Quantization(self.topo, self.geom)

    def test_josephson_vector_presence(self):
        """Ensure Josephson terms are correctly identified."""
        # 1. Check vector_JJ dimensions
        # Should have 1 column (1 Junction)
        self.assertEqual(self.quant.vector_JJ.shape[1], 1)
        
        # 2. Check vector is not zero (it projects onto the circuit variables)
        self.assertFalse(np.allclose(self.quant.vector_JJ, 0), 
                         "Vector JJ should not be zero for a connected junction")

    def test_hamiltonian_structure(self):
        """Check that Hamiltonian matrices are generated."""
        # Just checking existence and basic shape of the final result matrices
        H_phiq = self.quant.FS_quadratic_hamiltonian_phiq
        
        # Should be square
        self.assertEqual(H_phiq.shape[0], H_phiq.shape[1])
        
        # Should be complex128 or float (usually complex due to basis change)
        self.assertTrue(np.iscomplexobj(H_phiq) or np.issubdtype(H_phiq.dtype, np.floating))

if __name__ == '__main__':
    unittest.main()