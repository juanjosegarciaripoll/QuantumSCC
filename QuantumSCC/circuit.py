"""
circuit.py contains the classes for the circuit and their properties
"""

from typing import Any, Sequence, Optional, Union

import numpy as np
from scipy.linalg import null_space

from .elements import (
    Capacitor,
    Inductor,
    Junction
)

from .elements import Junction

from .algebra import *

Edge = tuple[int, int, object]

class Circuit:
    """
    Class that contains circuit properties.


    Parameters
    ----------
        elements:
            A dictionary that contains the circuit's elements at each branch
            of the circuit.

    """

    elements: list[Edge]
    no_elements: int
    node_dictionary: dict[Any, int]

    def __init__(self, elements: list[Edge]) -> None:
        """Define a circuit from a list of edges and circuit elements."""

        nodes = set([a for a, _, _ in elements] + [b for _, b, _ in elements])
        node_dictionary = {a: i for i, a in enumerate(nodes)}

        no_JJ, no_Capacitors, no_Inductors = 0, 0, 0

        complete_elements = []
        for a, b, elt in elements:
            if isinstance(elt, Junction) == True:
                complete_elements.append([node_dictionary[a], node_dictionary[b], elt])
                no_JJ += 1
        for a, b, elt in elements:
            if isinstance(elt, Junction) == True:
                complete_elements.append([node_dictionary[a], node_dictionary[b], elt.cap])
                no_Capacitors += 1
            elif isinstance(elt, Capacitor) == True:
                complete_elements.append([node_dictionary[a], node_dictionary[b], elt])
                no_Capacitors += 1
        for a, b, elt in elements:
            if isinstance(elt, Inductor) == True:
                complete_elements.append([node_dictionary[a], node_dictionary[b], elt])
                no_Inductors += 1

        self.elements = complete_elements

        self.no_JJ, self.no_Capacitors, self.no_Inductors = no_JJ, no_Capacitors, no_Inductors
        self.no_elements = len(self.elements)
        self.node_dict = node_dictionary
        self.no_nodes = len(node_dictionary)

        # [Construct F matrix from Sect. IIA]
        self.Fcut, self.Floop, self.F, self.K, self.no_reduced_compact_flux = self.Kirchhoff()

        # [Construct the differential form according to Sect. IIB]
        # [omega_2B is not needed, can be eliminated]
        self.omega_2B, self.omega_symplectic, self.V, self.no_independent_variables, self.no_final_compact_flux = self.omega_function()

        # [Construct the Hamiltonian according to the end of Sect. IIB and
        #  eliminate variables without "dynamic"; outputs Eq. (19)]
        self.quadratic_hamiltonian, self.vector_JJ = self.classical_hamiltonian_function()

        # [Section III, diagonalization of the quadratic part that is identified with oscillators]
        self.extended_quantum_hamiltonian, self.T, self.G = self.extended_hamiltonian_quantization()

        # [Adds the nonlinear part to the effective model, to treat junctions. Incomplete.]
        self.FS_quadratic_hamiltonian_phiq, self.FS_basis_change_phiq, self.final_vector_JJ_phiq, self.FS_quadratic_hamiltonian_an, self.FS_basis_change_an, self.final_vector_JJ_an = self.total_hamiltonian_quantization()


    def Kirchhoff(self):
        """
        Contructs the total Kirchhoff matrix F of the circuit and its kernel K.

        Returns
        ----------
        F_cut:
            Kirchhoff matrix with respect to the Kirchhoff current law (KCL).
        F_loop:
            Kirchhoff matrix with respect to the Kirchhoff voltage law (KVL).
        F:
            Total Kirchhoff matrix.
        K:
            Kernel of the total Kirchhoff matrix.
        """
        
        # Preallocate the F_cut matrix
        Fcut = np.zeros((self.no_nodes, self.no_elements))

        # Construct the F_cut matrix according to KCL
        for n_edge, (orig_node, dest_node, _) in enumerate(self.elements):
            Fcut[orig_node, n_edge] = -1
            Fcut[dest_node, n_edge] = +1

        Fcut, order = GaussJordan(Fcut)
        Fcut = reverseGaussJordan(remove_zero_rows(Fcut))

        # As we express Fcut = [1, A], construct Floop as Floop = [-A.T, 1]
        n = len(Fcut)
        A = Fcut[:, n:]
        Floop = np.hstack((-A.T, np.eye(A.shape[1])))

        # Reorder Fcut and Floop to have the same order as the array elements
        Fcut = Fcut[:, np.argsort(order)]
        Floop = Floop[:, np.argsort(order)]

        # Construct the full Kirchhoff matrix: F = [[Floop, 0], [0, Fcut]]
        F = np.block(
            [
                [Floop, np.zeros((Floop.shape[0], Fcut.shape[1]))],
                [np.zeros((Fcut.shape[0], Floop.shape[1])), Fcut],
            ]
        )

        # Construct Floop Kernel, Kloop, taking into account S1 and R variables
        # - Define the number of variables of each subspace
        no_initial_compact_flux_variables = self.no_JJ + self.no_Capacitors
        no_initial_extended_flux_variables = self.no_Inductors

        # - Construct the kernel of the compact subspace
        Floop_S = Floop[:, :no_initial_compact_flux_variables]
        Kloop_S = null_space(Floop_S)
        
        Kloop_S = np.vstack((Kloop_S, np.zeros((no_initial_extended_flux_variables, Kloop_S.shape[1])))) 

        no_reduced_compact_flux= Kloop_S.shape[1] # Calculate the number of compact fluxes.

        # - Construct the full space kernel, Kloop
        Kloop_aux = Fcut.T

        if Kloop_S.shape[1] == 0:
            Kloop = Kloop_aux
        else:
            Kloop = np.block([Kloop_S, Kloop_aux])
            Kloop = GS_algorithm(Kloop, normal=True, delete_zeros=True)
        
        # - Detect and simplify compact flux variables without dynamics
        #   [Investigate if we can, instead of doing this, eliminate the rows
        #    of dependent variables]
        if Kloop_S.shape[1] > 1:
            proportional_rows_Kloop = proportional_rows(Kloop[:no_initial_compact_flux_variables, :])
            
            if len(proportional_rows_Kloop) > 0:

                for _, rows_group in enumerate(proportional_rows_Kloop):
                    Kloop = Gauge_variable_symplification(Kloop, rows_group[0], rows_group[0])
        
        # Construct Fcut Kernel, Kcut
        Kcut = Floop.T

        # Construct the total kernel, K
        K = np.block(
            [
                [Kloop, np.zeros((Kloop.shape[0], Kcut.shape[1]))],
                [np.zeros((Kcut.shape[0], Kloop.shape[1])), Kcut],
            ]
        )

        # Make sure K is correct
        assert K.shape[1] == F.shape[1] - np.linalg.matrix_rank(K), "There is an error in the construction of the Kernel"
        assert np.allclose(F @ K, np.zeros((F.shape[0], K.shape[1]))) == True, "There is an error in the construction of the Kernel"

        return Fcut, Floop, F, K, no_reduced_compact_flux
    

    def omega_function(self):
        """
        Given the Lagrangian of the circuit: Lagrangian = omega - energy. It calculates the symplectic form of 
        the two-form omega and the basis change matrix.
        
        Returns
        ----------
        omega_2B:
            Matrix expression of the two-form omega.
        omega_symplectic:
            Symplectic expression of the two-form omega.
        V:
            Basis change matrix that transform omega to its symplectic form.
        number_of_pairs:
            Number of pairs of non-zero conjugate eigenvalues the two-form omega has. 
        """
        # Obtain omega_2B matrix
        omega_2B = np.zeros((2 * self.no_elements, 2 * self.no_elements))
        for i, elem in enumerate(self.elements):

            if isinstance(elem[2], Junction) == True:
                omega_2B[i, i + self.no_elements] = 0.5
                omega_2B[i + self.no_elements, i] = -0.5

            if isinstance(elem[2], Capacitor) == True:
                omega_2B[i, i + self.no_elements] = -0.5
                omega_2B[i + self.no_elements, i] = 0.5

            elif isinstance(elem[2], Inductor) == True:
                omega_2B[i, i + self.no_elements] = 0.5
                omega_2B[i + self.no_elements, i] = -0.5

        # Obtain omega matrix in the Kirchhoff equations basis
        omega_non_symplectic = self.K.T @ omega_2B @ self.K

        # Obtain the symplectic form of the omega matrix and the basis change matrix
        omega_symplectic, V, no_final_compact_flux = omega_symplectic_transformation(omega_non_symplectic,  no_compact_flux_variables=self.no_reduced_compact_flux, no_flux_variables = self.Fcut.shape[0])

        # Remove the zeros columns and rows from omega_symplectic
        # [The zeros introduced by the dependent variables in K appear as zeros
        #  in the matrix, that can be identified and eliminated. By the construction
        #  of "K" the dependent variables must have ended at the end (check!)]
        no_independent_variables = np.linalg.matrix_rank(omega_symplectic)
        omega_symplectic = omega_symplectic[:no_independent_variables, :no_independent_variables]

        assert no_final_compact_flux <= no_independent_variables//2, \
            "There is an error, the number of compact fluxes must me equal or smaller than the number of total fluxes"

        return omega_2B, omega_symplectic, V, no_independent_variables, no_final_compact_flux

    def classical_hamiltonian_function(self):
        """
        Given the Lagrangian of the circuit: Lagrangian = omega - energy. It constructs the symplified 
        quadratic Hamiltonian matrices from the energy function of the Lagrangian.

        Returns
        ----------
        quadratic_hamiltonian
            [Returns the matrix of the quadratic term in Eq. (19)]
        vector_JJ
            [Returns the vector 2*pi*v^T*K*V on which the cosine arguments
            are projected.]
        """

        # Calculate the initial quadratic total energy function matrix (prior to the change of variable given by the Kirchhoff's equtions)
        quadratic_energy = np.zeros((2 * self.no_elements, 2 * self.no_elements))
        for i, elem in enumerate(self.elements):

            if isinstance(elem[2], Inductor) == True:
                inductor = elem[2]
                quadratic_energy[i, i] = 2 * inductor.energy() # Energy of the inductor in GHz by default

            elif isinstance(elem[2], Capacitor) == True:
                capacitor = elem[2]
                quadratic_energy[i + self.no_elements, i + self.no_elements] = 2 * capacitor.energy() # Energy of the capacitor in GHz by default

        # Calculate the quadratic energy function matrix after the change of variables given by the Kirchhoff's equations
        quadratic_energy_after_Kirchhoff = self.K.T @ quadratic_energy @ self.K

        # Calculate the quadratic energy function matrix after the change of variables given by the symplectic form of omega
        quadratic_energy_symplectic_basis = self.V.T @ quadratic_energy_after_Kirchhoff @ self.V

        # Construct the initial vectors of the Josephson Juntion energy such that E = -Ej cos(vector.T @ R)
        vector_JJ = np.empty((quadratic_energy.shape[0], 0))
        for i, elem in enumerate(self.elements):
            if isinstance(elem[2], Junction) == True:
                aux = np.zeros((quadratic_energy.shape[0], 1))
                aux[i,0] = 1
                vector_JJ = np.hstack((vector_JJ, aux))

        # Calculate the JJ vector under the change of variables given by the Kirchhoff's equations and the symplectic form of omega
        vector_JJ = self.V.T @ self.K.T @ vector_JJ

        # Verify, JJ vector consider only dynamical variables and delete the entries corresponding to the non dynamical variables
        if  np.allclose(vector_JJ[self.no_independent_variables:,:], 0) == False:
            raise ValueError("The Energy of the Josephson Junction depends on non-dynamical variables. We cannot solve the circuit.")
        
        vector_JJ = vector_JJ[:self.no_independent_variables,:]

        # If the size of the new quadratic_energy_symplectic_basis matrix is equal to self.no_independent_variables, this matrix is the Hamiltonian
        if quadratic_energy_symplectic_basis.shape[0] == self.no_independent_variables:
            quadratic_hamiltonian = quadratic_energy_symplectic_basis

        # If the previous condition does not happend, we need to solve d(Total_energy_symplectic_basis)/dw = 0
        else:

            # Decompose thequadratic_energy_symplectic_basis matrix into 4 blocks: ((TEF_11, TEF_12);(TEF_21, TEF_22)), according to
            # the number of indpendent variables, self.no_independent_variables
            TEF_11 = quadratic_energy_symplectic_basis[:self.no_independent_variables, :self.no_independent_variables]
            TEF_12 = quadratic_energy_symplectic_basis[:self.no_independent_variables, self.no_independent_variables:]
            TEF_21 = quadratic_energy_symplectic_basis[self.no_independent_variables:, :self.no_independent_variables]
            TEF_22 = quadratic_energy_symplectic_basis[self.no_independent_variables:, self.no_independent_variables:]

            assert np.allclose(TEF_12, TEF_21.T) == True, "There is an error in the decomposition of the total energy function matrix in blocks"

            # Verify that the equation dH/dw = 0 has a solution by testing that TEF_22 has a inverse form
            try: 
                TEF_22_inv = pseudo_inv(TEF_22, tol = 1e-15)
            except np.linalg.LinAlgError:
                raise ValueError("There is no solution for the equation dH/dw = 0. The circuit does not present Hamiltonian dynamics.")

            # If there is solution, calculate the final matrix expression for the total energy function, which is the Hamiltonian
            quadratic_hamiltonian = TEF_11 - TEF_12 @ TEF_22_inv @ TEF_21


        # Verify the resulting quadratic Hamiltonian is block diagonal and symmetric
        assert np.allclose(quadratic_hamiltonian[quadratic_hamiltonian.shape[0]:, :quadratic_hamiltonian.shape[1]], 0) and \
            np.allclose(quadratic_hamiltonian[:quadratic_hamiltonian.shape[0], quadratic_hamiltonian.shape[1]:], 0), \
            'The classical Hamiltonian matrix must be block diagonal. There could be an error in the construction of the basis change matrix V'
        
        assert np.allclose(quadratic_hamiltonian.T, quadratic_hamiltonian), "Something goes wrong. Quadratic Hamiltonian matrix must be symmetric."

        # Ensure there are no compact fluxes in the quadratic Hamiltonian
        assert np.allclose(quadratic_hamiltonian[:self.no_final_compact_flux, :self.no_final_compact_flux], 0), \
            "Something goes wrong. No compact fluxes should appear in the quadratic Hamiltonian."

        # Returns
        return quadratic_hamiltonian, vector_JJ
    

    def extended_hamiltonian_quantization(self):
        """
        Calculates the extended quantum Hamiltonian in its canonical form.
        
        Returns
        ----------
        extended_quantum_hamiltonian:
            Cannical matrix expression of the extended quantum Hamiltonian.
        T:
            Basis change matrix that brings the Hamiltonian to its quantum canonical form.
        G:
            Basis change matrix that perform the second quantization of the Hamiltonian.
        """
        
        # Define the extended quadratic Hamiltonian
        extended_flux_indexes = np.arange(self.no_final_compact_flux, self.no_independent_variables//2)
        extended_charge_indexes = np.arange(self.no_final_compact_flux + self.no_independent_variables//2, self.no_independent_variables)
        extended_indexes = np.block([extended_flux_indexes, extended_charge_indexes])

        extended_quadratic_hamiltonian = self.quadratic_hamiltonian[np.ix_(extended_indexes, extended_indexes)]
        extended_dimension = extended_quadratic_hamiltonian.shape[0]

        # Get the quantum canonical Hamiltonian and the basis change matrix
        J = np.block([[np.zeros((extended_dimension//2, extended_dimension//2)), np.eye(extended_dimension//2)],
                      [-np.eye(extended_dimension//2), np.zeros((extended_dimension//2, extended_dimension//2))]])
        
        dynamical_matrix = J @ extended_quadratic_hamiltonian
        _, T = symplectic_transformation(dynamical_matrix, no_flux_variables=extended_quadratic_hamiltonian.shape[0]//2)
        extended_canonical_hamiltonian = T.T @ extended_quadratic_hamiltonian @ T

        # Proceed with the second quantization: Express the quantum Hamiltonian in the ladder operators basis.
        I = np.eye(len(extended_canonical_hamiltonian)//2)
        G = (1 / np.sqrt(2)) * np.block([[I, I], [-1j * I, 1j * I]])

        extended_quantum_hamiltonian = np.conj(G.T) @ extended_canonical_hamiltonian @ G

        # Verify the resulting Hamiltonian in the ladder operators basis is equal to the canonical Hamiltonian
        assert np.allclose(extended_quantum_hamiltonian, extended_canonical_hamiltonian), \
        "The matrix expression for the Hamiltonian in the ladder operators basis must be the same as the canonical Hamiltonian matrix."

        return extended_quantum_hamiltonian, T, G
    

    def total_hamiltonian_quantization(self):
        # [Incomplete version (maybe) of the projection of the nonlinear part onto
        #  the charge basis, for later diagonalization.]

        # Define the compact quadratic Hamiltonian 
        compact_flux_indexes = np.arange(0, self.no_final_compact_flux)
        compact_charge_indexes = np.arange(self.no_independent_variables//2, self.no_independent_variables//2 + self.no_final_compact_flux)
        compact_indexes = np.block([compact_flux_indexes, compact_charge_indexes])

        compact_quadratic_hamiltonian = self.quadratic_hamiltonian[np.ix_(compact_indexes, compact_indexes)]
        compact_dimension = compact_quadratic_hamiltonian.shape[0]

        # Diagonalize the compact quadratic Hamiltonian
        eigval, eigvec = np.linalg.eig(compact_quadratic_hamiltonian)
        sorted_indexes = np.argsort(np.abs(eigval))
        eigval = eigval[sorted_indexes]
        C = eigvec[:, sorted_indexes]

        diagonal_compact_quadratic_hamiltonian = C.T @ compact_quadratic_hamiltonian @ C

        # Define the vector of the JJ energy function
        vector_JJ = self.vector_JJ


        # Construct the full space basis change matrix for the flux-charge variables
        total_dimension = self.quadratic_hamiltonian.shape[0]
        T = self.T

        FS_basis_change_phiq = np.zeros((total_dimension, total_dimension), dtype=complex)

        FS_basis_change_phiq[:self.no_final_compact_flux, :self.no_final_compact_flux] = C[:C.shape[0]//2, :C.shape[1]//2]
        FS_basis_change_phiq[self.no_independent_variables//2:self.no_final_compact_flux + self.no_independent_variables//2, self.no_independent_variables//2:self.no_final_compact_flux + self.no_independent_variables//2] = C[C.shape[0]//2:, C.shape[1]//2:]
        
        FS_basis_change_phiq[self.no_final_compact_flux:self.no_independent_variables//2, self.no_final_compact_flux:self.no_independent_variables//2] = T[:T.shape[0]//2, :T.shape[1]//2]
        FS_basis_change_phiq[self.no_final_compact_flux:self.no_independent_variables//2, self.no_final_compact_flux + self.no_independent_variables//2:] = T[:T.shape[0]//2, T.shape[1]//2:]
        FS_basis_change_phiq[self.no_final_compact_flux + self.no_independent_variables//2:, self.no_final_compact_flux:self.no_independent_variables//2] = T[T.shape[0]//2:, :T.shape[1]//2]
        FS_basis_change_phiq[self.no_final_compact_flux + self.no_independent_variables//2:, self.no_final_compact_flux + self.no_independent_variables//2:] = T[T.shape[0]//2:, T.shape[1]//2:]

        # Construct the Full space almost diagonalized quadratic Hamiltonian for the flux-charge variables
        FS_quadratic_hamiltonian_phiq = np.conj(FS_basis_change_phiq.T) @ self.quadratic_hamiltonian @ FS_basis_change_phiq

        # Construct the final vector of the JJ energy function for the flux-charge variables
        final_vector_JJ_phiq = FS_basis_change_phiq.T @ vector_JJ


        # Construct the full space basis change matrix for the ladder operators, number-phase variables
        TG = self.T @ self.G

        FS_basis_change_an = np.zeros((total_dimension, total_dimension), dtype=complex)

        FS_basis_change_an[:self.no_final_compact_flux, :self.no_final_compact_flux] = C[:C.shape[0]//2, :C.shape[1]//2]
        FS_basis_change_an[self.no_independent_variables//2:self.no_final_compact_flux + self.no_independent_variables//2, self.no_independent_variables//2:self.no_final_compact_flux + self.no_independent_variables//2] = C[C.shape[0]//2:, C.shape[1]//2:]
        
        FS_basis_change_an[self.no_final_compact_flux:self.no_independent_variables//2, self.no_final_compact_flux:self.no_independent_variables//2] = TG[:TG.shape[0]//2, :TG.shape[1]//2]
        FS_basis_change_an[self.no_final_compact_flux:self.no_independent_variables//2, self.no_final_compact_flux + self.no_independent_variables//2:] = TG[:TG.shape[0]//2, TG.shape[1]//2:]
        FS_basis_change_an[self.no_final_compact_flux + self.no_independent_variables//2:, self.no_final_compact_flux:self.no_independent_variables//2] = TG[TG.shape[0]//2:, :TG.shape[1]//2]
        FS_basis_change_an[self.no_final_compact_flux + self.no_independent_variables//2:, self.no_final_compact_flux + self.no_independent_variables//2:] = TG[TG.shape[0]//2:, TG.shape[1]//2:]

        # Construct the Full space almost diagonalized quadratic Hamiltonian for the ladder operators, number-phase variables
        FS_quadratic_hamiltonian_an = np.conj(FS_basis_change_an.T) @ self.quadratic_hamiltonian @ FS_basis_change_an

        # Construct the final vector of the JJ energy function for the ladder operators, number-phase variables
        final_vector_JJ_an = FS_basis_change_an.T @ vector_JJ

        return FS_quadratic_hamiltonian_phiq, FS_basis_change_phiq, final_vector_JJ_phiq, FS_quadratic_hamiltonian_an, FS_basis_change_an, final_vector_JJ_an

        


    #######################################################################################################
    ################################### PRINT DIAGONALIZED HAMILTONIAN ####################################
    #######################################################################################################


    def diagonal_harmonic_Hamiltonian_expression(
            self, 
            precision: int = 3,
    ):
        """
        Print out the diagonalized Hamiltonian. 

        Parameters
        ----------
            precision: int
                Precision of the printing information. By default it going to print with 3 decimals places of precision
        """

        print('----------------------------------------------------------------------')

        # Print the diagonalized Hamiltonian
        extended_hamiltonian = self.extended_quantum_hamiltonian.real
        print(f'Diagonalized quantum Hamiltonian:')
        print(f'H/ℏ = ', end=" ")

        for i in range(len(extended_hamiltonian)//2):
            if i != len(extended_hamiltonian)//2 - 1:
                print(f'{extended_hamiltonian[i,i]:.{precision}f} GHz · (a\u2020_{i+1} a_{i+1}) + ', end=" ")
            else:
                print(f'{extended_hamiltonian[i,i]:.{precision}f} GHz · (a\u2020_{i+1} a_{i+1})')
            
        print('----------------------------------------------------------------------')
    
    

    def Hamiltonian_expression(
            self, 
            precision: int = 3,
            tol: float = 1e-14
    ):
        """
        Print out the Hamiltonian. 

        Parameters
        ----------
            precision: int
                Precision of the printing information. By default it going to print with 3 decimals places of precision
            tol: float
                Tolerance below which a number is considered zero. By default, it is 1e-14.
        """

        # Define the matrices
        quantum_quadratic_hamiltonian = self.FS_quadratic_hamiltonian_phiq.real
        vector_JJ = self.final_vector_JJ_phiq

        # Define dimensional tools
        no_flux_variables = quantum_quadratic_hamiltonian.shape[0]//2
        no_compact_fluxes = self.no_final_compact_flux
        no_JJ = self.no_JJ

        print('----------------------------------------------------------------------')

        # Print the  Hamiltonian
        print(f'Quantum Hamiltonian:')
        print(f'H/ℏ (GHz) =', end=" ")

        # Print the extended Hamiltonian
        for i in range(no_compact_fluxes, no_flux_variables):
            if np.abs(quantum_quadratic_hamiltonian[i,i]) > 1e-14:
                print(f'+ {quantum_quadratic_hamiltonian[i,i]:.{precision}f} [(\u03D5_e{i-no_compact_fluxes+1})^2 + (n_e{i-no_compact_fluxes+1})^2]', end=" ")
        
        # Print interaction Hamiltonian
        for i in range(no_flux_variables, 2*no_flux_variables):
            for j in range(no_flux_variables, 2*no_flux_variables):
                if np.abs(quantum_quadratic_hamiltonian[i,j]) > 1e-14 and i > j:
                    print(f' + {(2 * quantum_quadratic_hamiltonian[i,j]):.{precision}f} n_e{i-no_flux_variables-no_compact_fluxes+1} n_c{j-no_flux_variables+1}', end=" ")

        # Print non-linear Hamiltonian
        for i in range(no_compact_fluxes):
            if np.abs(quantum_quadratic_hamiltonian[i+no_flux_variables, i+no_flux_variables]) > 1e-14:
                print(f' + {quantum_quadratic_hamiltonian[i+no_flux_variables,i+no_flux_variables]:.{precision}f} (n_c{i+1})^2', end=" ")

        junction_energy = np.zeros(no_JJ)
        for i, elem in enumerate(self.elements):
            if isinstance(elem[2], Junction) == True:
                junction = elem[2]
                junction_energy[i] = junction.value()
        
        for i in range(no_JJ):
            if i != no_JJ-1:
                print(f' - {junction_energy[i]:.{precision}f} cos(v_{i+1} \u03BE)', end=" ")
            else:
                print(f' - {junction_energy[i]:.{precision}f} cos(v_{i+1} \u03BE)')
        if no_JJ == 0:
            print('')

        print('')

        np.set_printoptions(precision=precision)
        print(f'Vectors v:')
        for i in range(vector_JJ.shape[1]):
            print(f'v_{i+1} = {(vector_JJ[:,i].real).T}')

        print('')

        print(f'Variable vectors \u03BE:')
        print(f'\u03BEᵀ = (', end=" ")
        for i in range(2*no_flux_variables):
            if i < no_compact_fluxes:
                print(f'\u03D5_c{i+1}', end=" ")
            elif no_compact_fluxes <= i < no_flux_variables:
                print(f' \u03D5_e{i-no_compact_fluxes+1}', end=" ")
            elif no_flux_variables <= i < no_flux_variables + no_compact_fluxes:
                print(f' n_c{i-no_flux_variables+1}', end=" ")
            elif  no_flux_variables + no_compact_fluxes <= i <= 2*no_flux_variables-1:
                print(f' n_e{i-no_compact_fluxes-no_flux_variables+1}', end=" ")
        print(f')')
        print(f'')

        # Return the opological behavior of each operator
        print(f'Operator subscripts explanation:')
        print(f' - Subindex e indicates that the operator belongs to the extended flux subspace and their conjugated charges')
        print(f' - Subindex c indicates that the operator belongs to the compact flux subspace and their conjugated charges')
        print('')

        # Give the information about number-phase operators
        print(f'Relation between number-phase operators and flux-charge operators:')
        print(f' - n = Q/(2e)')
        print(f' - \u03D5 = 2\u03C0 \u03C6/(\u03C6_0)')
        print('----------------------------------------------------------------------')



    

        
                

    



        





   






        
        
