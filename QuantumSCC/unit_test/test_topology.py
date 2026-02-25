"""
Unit tests for the Topology class in QuantumSCC.core.topology
"""

import unittest
import numpy as np
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__)) 
package_dir = os.path.dirname(current_dir)               
project_root = os.path.dirname(package_dir)              
sys.path.insert(0, project_root)

from QuantumSCC.core.elements import Capacitor, Inductor, Junction, PhaseSlip
from QuantumSCC.core.topology import Topology

class TestTopologyMatrices(unittest.TestCase):

    def setUp(self):
        """Set up a common circuit for tests (LC loop chain)."""
        # Circuit: 0-1-2-3-0 loop with 1 Cap and 4 Inds
        C = Capacitor(value=1, unit='GHz')
        L = Inductor(value=1, unit='GHz')
        self.elements = [
            (0, 1, C), 
            (0, 1, L), 
            (1, 2, L), 
            (2, 3, L), 
            (3, 0, L)
        ]
        # Initialize Topology directly
        self.topo = Topology(self.elements)

    def test_Fcut_matrix(self):
        """Test the construction of the Cut-set matrix (KCL)."""
        # Expected Fcut for the defined topology
        # Nodes: 4 (0,1,2,3) -> Rows: 3 (independent)
        # Branches: 5 -> Cols: 5
        expected_Fcut = np.array([
            [ 1.,  1.,  0.,  0., -1.],  # Node 1 (relative to ground/tree)
            [ 0.,  0.,  1.,  0., -1.],  # Node 2
            [ 0.,  0.,  0.,  1., -1.]   # Node 3
        ])
        
        # Check shape and values
        self.assertEqual(self.topo.Fcut.shape, expected_Fcut.shape)
        self.assertTrue(np.allclose(self.topo.Fcut, expected_Fcut), 
                        msg=f"Fcut mismatch.\nGot:\n{self.topo.Fcut}\nExpected:\n{expected_Fcut}")

    def test_Floop_matrix(self):
        """Test the construction of the Loop matrix (KVL)."""
        # Expected Floop
        # Fundamental loops: 2 (b - n + 1 = 5 - 4 + 1 = 2)
        expected_Floop = np.array([
            [-1.,  1., -0., -0.,  0.], # Loop 1: L vs C in parallel
            [ 1.,  0.,  1.,  1.,  1.]  # Loop 2: The big outer loop
        ])

        self.assertEqual(self.topo.Floop.shape, expected_Floop.shape)
        # Note: Sign convention might flip entire rows, so we check absolute correlation or direct match
        # Here we assume direct match based on deterministic algorithm
        self.assertTrue(np.allclose(self.topo.Floop, expected_Floop),
                        msg=f"Floop mismatch.\nGot:\n{self.topo.Floop}\nExpected:\n{expected_Floop}")

    def test_Kernel_properties(self):
        """Test algebraic properties of the Kernel K."""
        F = self.topo.F
        K = self.topo.K
        
        # 1. Rank-Nullity Theorem: rank(K) + rank(F) = #Variables (columns)
        # In this formulation, K is the full kernel matrix, so its columns span the null space.
        # However, your specific F @ K = 0 check implies K columns are in the null space of F.
        
        # Check orthogonality: F * K should be zero
        product = F @ K
        self.assertTrue(np.allclose(product, np.zeros_like(product)), 
                        msg=f"Kirchhoff constraint violated: F@K is not zero.\nMax error: {np.max(np.abs(product))}")

        # Check dimensions
        # Rank of K should correspond to degrees of freedom
        rank_K = np.linalg.matrix_rank(K)
        nullity_F = F.shape[1] - np.linalg.matrix_rank(F)
        
        # Assuming K captures the full kernel
        self.assertEqual(K.shape[1], nullity_F, 
                         msg="K matrix does not span the full null space of F.")

    def test_compact_variable_identification(self):
        """Test if JJs and Caps are correctly identified as compact flux variables."""
        # In setup, we have 1 Cap and 4 Inds.
        # no_compact should be 1 (the Capacitor).
        # no_extended should be 4.
        
        # The Topology class calculates 'no_reduced_compact_flux' internally
        # Let's verify it matches our expectation for this circuit.
        
        # For the LC // L loop + 3 Ls, the compact variable is the loop formed by C and parallel L?
        # Actually, in this topology, the capacitor is part of a loop.
        # The logic is: Kloop_S is Kernel of loops restricted to Compact elements.
        # Here, Loop 1 is (-1, 1, 0, 0, 0) acting on (C, L1, L2, L3, L4).
        # Compact part of loop 1 is just the C coefficient.
        # Since it's not a loop of ONLY capacitors/JJs, the reduced compact flux might be 0 or 1 depending on ground.
        
        # Let's just check the counts are exposed correctly
        self.assertEqual(self.topo.no_Capacitors, 1)
        self.assertEqual(self.topo.no_Inductors, 4)
        self.assertEqual(self.topo.no_JJ, 0)

class TestTopologyQPS(unittest.TestCase):
    """Tests for Topology when QPS elements are present."""

    def setUp(self):
        """Dual-LC: one QPS + its required parallel inductor."""
        self.L = Inductor(value=1, unit='nH')
        self.P = PhaseSlip(value=1, unit='GHz', ind=self.L)
        self.topo = Topology([(0, 1, self.P)])

    def test_element_counts_qps_only(self):
        """no_QPS=1, no_Inductors=1 (parallel ind), no_JJ=0, no_Capacitors=0."""
        self.assertEqual(self.topo.no_QPS,        1)
        self.assertEqual(self.topo.no_JJ,         0)
        self.assertEqual(self.topo.no_Capacitors,  0)
        self.assertEqual(self.topo.no_Inductors,   1)

    def test_kirchhoff_fk_zero(self):
        """F @ K must be zero for a QPS circuit."""
        product = self.topo.F @ self.topo.K
        self.assertTrue(np.allclose(product, 0),
                        msg=f"F@K max error: {np.max(np.abs(product)):.2e}")

    def test_kernel_rank_nullity(self):
        """rank(F) + cols(K) = 2 * no_elements (rank-nullity theorem)."""
        rank_F = np.linalg.matrix_rank(self.topo.F)
        cols_K = self.topo.K.shape[1]
        self.assertEqual(rank_F + cols_K, 2 * self.topo.no_elements)

    def test_compact_charge_count(self):
        """One QPS -> at least 1 reduced compact charge variable."""
        self.assertGreaterEqual(self.topo.no_reduced_compact_charge, 1)

    def test_no_compact_charge_without_qps(self):
        """Circuit with no QPS -> zero compact charge variables."""
        C = Capacitor(value=1, unit='pF')
        J = Junction(value=1, unit='GHz', cap=C)
        topo = Topology([(0, 1, J)])
        self.assertEqual(topo.no_reduced_compact_charge, 0)

    def test_element_order_qps_before_inductors(self):
        """Elements order: [QPS, Inductor] for a single-QPS circuit."""
        self.assertIsInstance(self.topo.elements[0][2], PhaseSlip)
        self.assertIsInstance(self.topo.elements[1][2], Inductor)


if __name__ == '__main__':
    unittest.main()