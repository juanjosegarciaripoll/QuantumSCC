"""
Unit tests for the Geometry class in QuantumSCC.core.geometry
"""

import unittest
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC.core.elements import Capacitor, Inductor
from QuantumSCC.core.topology import Topology
from QuantumSCC.core.geometry import Geometry

class TestSymplecticGeometry(unittest.TestCase):

    def setUp(self):
        """
        Set up the Geometry analysis for a known circuit.
        Circuit: LC loop + chain (Same as original test example)
        Topology: 0-1(C), 0-1(L), 1-2(L), 2-3(L), 3-0(L)
        """
        C = Capacitor(value=1, unit='GHz')
        L = Inductor(value=1, unit='GHz')
        
        elements = [
            (0, 1, C), 
            (0, 1, L), 
            (1, 2, L), 
            (2, 3, L), 
            (3, 0, L)
        ]
        
        self.topo = Topology(elements)
        self.geom = Geometry(self.topo)

    def test_omega_2B_construction(self):
        """Test the construction of the 2-Body symplectic form (omega_2B)."""
        expected_omega_2B = np.zeros((10, 10))
        
        # Capacitor at index 0
        expected_omega_2B[0, 5] = -0.5
        expected_omega_2B[5, 0] =  0.5
        
        # Inductors at indices 1, 2, 3, 4
        for i in range(1, 5):
            expected_omega_2B[i, i+5] =  0.5
            expected_omega_2B[i+5, i] = -0.5
            
        self.assertTrue(np.allclose(self.geom.omega_2B, expected_omega_2B),
                        msg="omega_2B matrix construction is incorrect.")

    def test_symplectic_reduction(self):
        """Test the reduced symplectic form (omega_symplectic)."""
        # We expect 1 dynamic mode -> 2x2 symplectic matrix
        expected_omega_symplectic = np.array([
            [ 0.,  1.],
            [-1.,  0.]
        ])
        
        # This checks that we correctly identified 2 dynamic variables
        self.assertEqual(self.geom.omega_symplectic.shape, expected_omega_symplectic.shape)
        self.assertTrue(np.allclose(self.geom.omega_symplectic, expected_omega_symplectic))

    def test_variables_count_and_V_shape(self):
        """Test independent variables count and V matrix dimensions."""
        # 1. Check Independent Variables (Dynamic Modes)
        # Should be 2 (1 mode pair phi, q)
        self.assertEqual(self.geom.no_independent_variables, 2)
        
        # 2. Check V Matrix Shape
        # V transforms from Kirchhoff Basis (K) to Canonical Basis.
        # K shape for this circuit is (10, 5).
        # So V must be square with respect to K's columns -> (5, 5).
        k_columns = self.topo.K.shape[1]
        self.assertEqual(self.geom.V.shape, (k_columns, k_columns))
        
        # 3. Optional: Verify V is part of a valid symplectic transformation
        # V.T @ Omega_NonSymplectic @ V = J (Canonical)
        # We just check dimensions here to keep it simple unit test
        self.assertEqual(self.geom.V.shape, (5, 5))

if __name__ == '__main__':
    unittest.main()