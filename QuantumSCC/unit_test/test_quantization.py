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

from QuantumSCC.core.elements import Capacitor, Inductor, Junction, PhaseSlip
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

class TestQuantizationQPS(unittest.TestCase):
    """Tests for the dual-transmon (QPS + Inductor) quantization."""

    def setUp(self):
        """Dual-LC: PhaseSlip(1 GHz) in parallel with Inductor(1 nH)."""
        self.L_nH = 1.0
        self.E_P  = 1.0
        L = Inductor(value=self.L_nH, unit='nH')
        P = PhaseSlip(value=self.E_P, unit='GHz', ind=L)
        self.topo  = Topology([(0, 1, P)])
        self.geom  = Geometry(self.topo)
        self.quant = Quantization(self.topo, self.geom)

        # Analytical E_L in code units: (Phi0/2pi)^2 / (2*L*hbar) / 1e9
        from QuantumSCC.utils import units as unt
        self.E_L_code = (unt.Phi0 / (2 * np.pi))**2 / (2 * self.L_nH * 1e-9 * unt.hbar) / 1e9

    def test_quadratic_hamiltonian_shape(self):
        """Dual-LC has 1 mode -> 2x2 quadratic Hamiltonian."""
        self.assertEqual(self.quant.quadratic_hamiltonian.shape, (2, 2))

    def test_flux_entry_matches_formula(self):
        """H[0,0] = 2 * E_L_code (inductive energy, factor 2 from unnormalized kernel)."""
        H = self.quant.quadratic_hamiltonian
        expected = 2.0 * self.E_L_code
        self.assertAlmostEqual(H[0, 0].real, expected, delta=expected * 1e-6)

    def test_charge_entry_zero(self):
        """H[1,1] = 0: compact charge has no quadratic energy."""
        H = self.quant.quadratic_hamiltonian
        self.assertAlmostEqual(abs(H[1, 1]), 0.0, delta=1e-10)

    def test_vector_qps_nonzero(self):
        """vector_QPS must not be zero — QPS couples to dynamics."""
        self.assertFalse(np.allclose(self.quant.vector_QPS, 0),
                         msg="vector_QPS is zero — QPS decoupled from dynamics")

    def test_vector_qps_single_column(self):
        """One QPS element -> vector_QPS has exactly 1 column."""
        self.assertEqual(self.quant.vector_QPS.shape[1], 1)

    def test_vector_jj_empty(self):
        """No JJ -> vector_JJ has 0 columns."""
        self.assertEqual(self.quant.vector_JJ.shape[1], 0)


if __name__ == '__main__':
    unittest.main()