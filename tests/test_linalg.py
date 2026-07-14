"""
Unit tests for linear algebra functions in QuantumSCC.utils.linalg
"""

import unittest

import numpy as np

from QuantumSCC.utils.linalg import (
    GaussJordan,
    omega_symplectic_transformation,
    reverseGaussJordan,
    symplectic_transformation,
)


class Test_Gauss_Jordan_method(unittest.TestCase):

    def test_direct_Gauss_Jordan(self):
        """Test the forward Gauss-Jordan elimination."""
        M_before_GJ = np.array([[-1., -1.,  0.,  0.,  1.,],
                                [ 1.,  1., -1.,  0.,  0.,],
                                [ 0.,  0.,  1., -1.,  0.,],
                                [ 0.,  0.,  0.,  1., -1.,]])
        
        M_after_GJ = np.array([[-1.,  0.,  0., -1.,  1.,],
                               [ 0., -1.,  0.,  0.,  1.,],
                               [ 0.,  0., -1.,  0.,  1.,],
                               [ 0.,  0.,  0.,  0.,  0.,]])
        
        M_test, _ = GaussJordan(M_before_GJ)

        self.assertTrue(np.allclose(M_test, M_after_GJ))

    def test_reverse_Gauss_Jordan(self):
        """Test the reverse Gauss-Jordan elimination (diagonalization)."""
        M_before_reverse_GJ = np.array([[-1.,  0.,  0., -1.,  1.],
                                        [ 0., -1.,  0.,  0.,  1.],
                                        [ 0.,  0., -1.,  0.,  1.]])
        
        M_after_reverse_GJ = np.array([[ 1.,  0.,  0.,  1., -1.],
                                       [ 0.,  1.,  0.,  0., -1.],
                                       [ 0.,  0.,  1.,  0., -1.]])
        
        M_test = reverseGaussJordan(M_before_reverse_GJ)
        
        self.assertTrue(np.allclose(M_test, M_after_reverse_GJ))

class Test_symplectic_form_function(unittest.TestCase):

    def test_symplectic_form_transformation(self):
        """Test the transformation of a generic antisymmetric matrix to canonical form."""
        matrix_before_transformation = np.array([[ 0.,  0.,  0.,  1., -1.],
                                                 [ 0.,  0.,  0.,  0., -1.],
                                                 [ 0.,  0.,  0.,  0.,  0.],
                                                 [-1.,  0.,  0.,  0.,  0.],
                                                 [ 1.,  1.,  0.,  0.,  0.]])
        
        matrix_after_transformation = np.array([[ 0.,  0.,  1.,  0.,  0.],
                                                [ 0.,  0.,  0.,  1.,  0.],
                                                [-1., -0.,  0.,  0.,  0.],
                                                [-0., -1.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0.]])
        
        # Note: no_compact_flux_variables=0, no_flux_variables=2 used as dummy params for
        # pure math test
        matrix_test, _, _, _ = omega_symplectic_transformation(
            matrix_before_transformation,
            no_compact_flux_variables=0,
            no_flux_variables=2
        )
        
        self.assertTrue(np.allclose(matrix_after_transformation, matrix_test))

    def test_symplectic_form_basis_change(self):
        """Test consistency: Canonical = V.T @ Original @ V."""
        matrix_before_transformation = np.array([[ 0.,  0.,  0.,  1., -1.],
                                                 [ 0.,  0.,  0.,  0., -1.],
                                                 [ 0.,  0.,  0.,  0.,  0.],
                                                 [-1.,  0.,  0.,  0.,  0.],
                                                 [ 1.,  1.,  0.,  0.,  0.]])
        
        canonical_matrix, canonical_basis_change, _, _ = omega_symplectic_transformation(
            matrix_before_transformation,
            no_compact_flux_variables=0,
            no_flux_variables=2
        )

        transform_check = (
            canonical_basis_change.T @ matrix_before_transformation @ canonical_basis_change
        )
        self.assertTrue(np.allclose(canonical_matrix, transform_check))

class Test_canonical_transformation_quadratic_hamiltonian(unittest.TestCase):

    def setUp(self):
        # Common data for these tests
        self.hamiltonian = np.array([[ 5.963e-01,  1.491e-01, -1.342e-16,  1.573e-17],
                                     [ 1.491e-01,  5.963e-01, -6.702e-17,  3.462e-17],
                                     [-1.342e-16, -6.702e-17,  5.963e-01, -1.491e-01],
                                     [ 1.573e-17,  3.462e-17, -1.491e-01,  5.963e-01]])
        
        self.J = np.block([[ np.zeros((2,2)), np.eye(2)], 
                           [-np.eye(2), np.zeros((2,2))]])

    def test_basis_change_matrix_T_dimensions(self):
        """Test dimensions of the symplectic diagonalizer T."""
        _, T = symplectic_transformation(self.J @ self.hamiltonian, no_flux_variables=2)
        
        self.assertEqual(T.shape[0], self.hamiltonian.shape[0])
        self.assertEqual(T.shape[1], self.hamiltonian.shape[1])

    def test_basis_change_matrix_T_symplectic(self):
        """Test that T is a symplectic matrix (preserves J)."""
        _, T = symplectic_transformation(self.J @ self.hamiltonian, no_flux_variables=2)
        
        # Symplectic condition: T.T @ J @ T = J
        self.assertTrue(np.allclose(self.J, T.T @ self.J @ T))

    def test_basis_change_matrix_T_real(self):
        """Test that T contains only real values (for real Hamiltonians)."""
        _, T = symplectic_transformation(self.J @ self.hamiltonian, no_flux_variables=2)

        self.assertTrue(np.allclose(T.imag, 0))

if __name__ == '__main__':
    unittest.main()