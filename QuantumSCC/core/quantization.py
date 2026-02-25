"""
core/quantization.py

Handles the construction of the classical and quantum Hamiltonians,
as well as the diagonalization of the harmonic subspace.
Corresponds to Sections III, IV, and V.
"""

import numpy as np
from .elements import Junction, Capacitor, Inductor, PhaseSlip
from ..utils.linalg import pseudo_inv, symplectic_transformation

class Quantization:
    def __init__(self, topology, geometry, debug: bool = False):
        """
        Initializes the quantization process using Topology and Geometry results.
        """
        self.debug = debug
        self.topo = topology
        self.geom = geometry

        # Construct the Hamiltonian according to the end of Sect. IIB
        self.quadratic_hamiltonian, self.vector_JJ, self.vector_QPS = self.classical_hamiltonian_function()

        # Section III, diagonalization of the quadratic part
        self.extended_quantum_hamiltonian, self.T, self.G = self.extended_hamiltonian_quantization()

        # Adds the nonlinear part to the effective model
        self.FS_quadratic_hamiltonian_phiq, self.FS_basis_change_phiq, \
        self.final_vector_JJ_phiq, self.final_vector_QPS_phiq, \
        self.FS_quadratic_hamiltonian_an, self.FS_basis_change_an, \
        self.final_vector_JJ_an, self.final_vector_QPS_an = self.total_hamiltonian_quantization()

    def classical_hamiltonian_function(self):
        """
        Given the Lagrangian of the circuit: Lagrangian = omega - energy. It constructs the symplified 
        quadratic Hamiltonian matrices from the energy function of the Lagrangian.
        """
        if self.debug:
            print("\n" + "-"*40)
            print("3. CLASSICAL HAMILTONIAN")
            print("-"*40)
            print("Eq (13) - Lagrangian: L = 1/2 Z_dot Omega Z - 1/2 Z E Z + NonLinear")
            print("Eq (18) - Eff. Hamiltonian: H = H_xx - H_xw H_ww^-1 H_wx")
            print("Eq (19) - Final Classical H: H(x) = 1/2 x^T H x - sum E_J cos(...)")

        # Calculate the initial quadratic total energy function matrix
        quadratic_energy = np.zeros((2 * self.topo.no_elements, 2 * self.topo.no_elements))
        for i, elem in enumerate(self.topo.elements):

            if isinstance(elem[2], Inductor):
                inductor = elem[2]
                quadratic_energy[i, i] = 2 * inductor.energy()

            elif isinstance(elem[2], Capacitor):
                capacitor = elem[2]
                quadratic_energy[i + self.topo.no_elements, i + self.topo.no_elements] = 2 * capacitor.energy()

        # Calculate the quadratic energy function matrix after Kirchhoff
        quadratic_energy_after_Kirchhoff = self.topo.K.T @ quadratic_energy @ self.topo.K

        # Calculate the quadratic energy function matrix after symplectic basis change
        quadratic_energy_symplectic_basis = self.geom.V.T @ quadratic_energy_after_Kirchhoff @ self.geom.V

        # Construct the initial vectors of the Josephson Junction energy (flux sector)
        vector_JJ = np.empty((quadratic_energy.shape[0], 0))
        for i, elem in enumerate(self.topo.elements):
            if isinstance(elem[2], Junction):
                aux = np.zeros((quadratic_energy.shape[0], 1))
                aux[i, 0] = 1
                vector_JJ = np.hstack((vector_JJ, aux))

        # Calculate the JJ vector under the change of variables
        vector_JJ = self.geom.V.T @ self.topo.K.T @ vector_JJ

        # Verify JJ vector considers only dynamical variables
        if np.allclose(vector_JJ[self.geom.no_independent_variables:, :], 0) == False:
            raise ValueError("The Energy of the Josephson Junction depends on non-dynamical variables. We cannot solve the circuit.")

        vector_JJ = vector_JJ[:self.geom.no_independent_variables, :]

        # Construct the initial vectors of the PhaseSlip energy (charge sector — dual to JJ)
        vector_QPS = np.empty((quadratic_energy.shape[0], 0))
        for i, elem in enumerate(self.topo.elements):
            if isinstance(elem[2], PhaseSlip):
                aux = np.zeros((quadratic_energy.shape[0], 1))
                aux[i + self.topo.no_elements, 0] = 1   # charge sector row
                vector_QPS = np.hstack((vector_QPS, aux))

        # Calculate the QPS vector under the change of variables
        vector_QPS = self.geom.V.T @ self.topo.K.T @ vector_QPS

        # Verify QPS vector considers only dynamical variables
        if vector_QPS.shape[1] > 0:
            if np.allclose(vector_QPS[self.geom.no_independent_variables:, :], 0) == False:
                raise ValueError("The Energy of the PhaseSlip element depends on non-dynamical variables. We cannot solve the circuit.")

        vector_QPS = vector_QPS[:self.geom.no_independent_variables, :]

        # If quadratic_energy_symplectic_basis size equals independent variables, it is the Hamiltonian
        if quadratic_energy_symplectic_basis.shape[0] == self.geom.no_independent_variables:
            quadratic_hamiltonian = quadratic_energy_symplectic_basis

        # Otherwise solve d(Total_energy)/dw = 0
        else:
            no_indep = self.geom.no_independent_variables
            TEF_11 = quadratic_energy_symplectic_basis[:no_indep, :no_indep]
            TEF_12 = quadratic_energy_symplectic_basis[:no_indep, no_indep:]
            TEF_21 = quadratic_energy_symplectic_basis[no_indep:, :no_indep]
            TEF_22 = quadratic_energy_symplectic_basis[no_indep:, no_indep:]

            assert np.allclose(TEF_12, TEF_21.T) == True, "There is an error in the decomposition of the total energy function matrix in blocks"

            try: 
                TEF_22_inv = pseudo_inv(TEF_22, tol = 1e-15)
            except np.linalg.LinAlgError:
                raise ValueError("There is no solution for the equation dH/dw = 0. The circuit does not present Hamiltonian dynamics.")

            quadratic_hamiltonian = TEF_11 - TEF_12 @ TEF_22_inv @ TEF_21


        # Verify the resulting quadratic Hamiltonian is block diagonal and symmetric
        assert np.allclose(quadratic_hamiltonian[quadratic_hamiltonian.shape[0]:, :quadratic_hamiltonian.shape[1]], 0) and \
            np.allclose(quadratic_hamiltonian[:quadratic_hamiltonian.shape[0], quadratic_hamiltonian.shape[1]:], 0), \
            'The classical Hamiltonian matrix must be block diagonal. There could be an error in the construction of the basis change matrix V'
        
        assert np.allclose(quadratic_hamiltonian.T, quadratic_hamiltonian), "Something goes wrong. Quadratic Hamiltonian matrix must be symmetric."

        # Ensure there are no compact fluxes in the quadratic Hamiltonian
        assert np.allclose(quadratic_hamiltonian[:self.geom.no_final_compact_flux, :self.geom.no_final_compact_flux], 0), \
            "Something goes wrong. No compact fluxes should appear in the quadratic Hamiltonian."

        if self.debug:
            print(f"Quadratic Hamiltonian shape: {quadratic_hamiltonian.shape}")
            print(f"Trace: {np.trace(quadratic_hamiltonian):.6f}")

        return quadratic_hamiltonian, vector_JJ, vector_QPS
    

    def extended_hamiltonian_quantization(self):
        """
        Calculates the extended quantum Hamiltonian in its canonical form.
        """
        if self.debug:
            print("\n" + "-"*40)
            print("4. QUANTIZATION")
            print("-"*40)
            print("Eq (20) - Quantum H: H_hat = 1/2 xi^T H xi - sum E_J cos(phi_j)")
            print("Eq (25) - Ladder Ops: Psi = G A")
            print("Eq (26) - Diagonal H_e: H_e = sum h_bar omega_j a_j^dagger a_j")

        # Define the extended quadratic Hamiltonian
        # Exclude both compact flux (JJ) and compact charge (QPS) variables.
        no_compact_flux   = self.geom.no_final_compact_flux
        no_compact_charge = self.geom.no_final_compact_charge
        no_indep          = self.geom.no_independent_variables
        no_flux           = no_indep // 2

        # Extended flux: flux variables that are neither JJ compact flux nor
        # QPS-inductor flux (the first no_compact_charge extended flux variables
        # pair with the compact charges and are NOT harmonic oscillator modes).
        extended_flux_indexes = np.arange(no_compact_flux + no_compact_charge, no_flux)
        # Extended charge: charge variables beyond the JJ-conjugate charges and
        # the QPS compact charges.
        extended_charge_indexes = np.arange(
            no_flux + no_compact_flux + no_compact_charge,
            no_indep
        )
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

        if self.debug:
            print(f"Extended Quantum H shape: {extended_quantum_hamiltonian.shape}")
            print(f"Modes found: {len(extended_quantum_hamiltonian)//2}")
            eigenvals = np.diagonal(extended_quantum_hamiltonian.real)
            print(f"Hamiltonian eigenvalues: {eigenvals}")

        return extended_quantum_hamiltonian, T, G
    

    def total_hamiltonian_quantization(self):
        # Define the compact quadratic Hamiltonian.
        # The compact sector includes both JJ compact flux pairs AND QPS compact charge pairs.
        # nc_total = nCF + nCC counts all compact (non-oscillator) pairs.
        no_compact_flux   = self.geom.no_final_compact_flux
        no_compact_charge = self.geom.no_final_compact_charge
        nc_total  = no_compact_flux + no_compact_charge
        no_indep  = self.geom.no_independent_variables
        no_flux   = no_indep // 2

        # Compact flux indices: JJ compact flux [0..nCF-1] + QPS-inductor flux [nCF..nc_total-1]
        compact_flux_indexes = np.arange(0, nc_total)
        # Compact charge indices: JJ-conjugate charges [nF..nF+nCF-1] + QPS compact charges [nF+nCF..nF+nc_total-1]
        compact_charge_indexes = np.arange(no_flux, no_flux + nc_total)
        compact_indexes = np.block([compact_flux_indexes, compact_charge_indexes])

        compact_quadratic_hamiltonian = self.quadratic_hamiltonian[np.ix_(compact_indexes, compact_indexes)]
        
        # Diagonalize flux and charge compact subblocks SEPARATELY.
        # Joint diagonalization fails when QPS compact charges (eigenvalue 0) and
        # JJ compact fluxes (eigenvalue 0) both appear, mixing the blocks incorrectly.
        #
        # Flux block A: JJ compact fluxes have eigenvalue 0, QPS-inductor fluxes have E_L.
        #   Ascending sort → JJ (0) first, QPS-inductor (E_L) last.
        # Charge block D: JJ-conjugate charges have E_C, QPS compact charges have eigenvalue 0.
        #   Descending sort → JJ (E_C) first, QPS compact charges (0) last.
        A = compact_quadratic_hamiltonian[:nc_total, :nc_total]
        eigval_A, eigvec_A = np.linalg.eig(A)
        sorted_A = np.argsort(np.abs(eigval_A))         # ascending: JJ zeros first
        C_flux = np.real(eigvec_A[:, sorted_A])

        D = compact_quadratic_hamiltonian[nc_total:, nc_total:]
        eigval_D, eigvec_D = np.linalg.eig(D)
        sorted_D = np.argsort(-np.abs(eigval_D))        # descending: JJ E_C first, QPS 0 last
        C_charge = np.real(eigvec_D[:, sorted_D])

        # Define the vectors of the JJ and QPS energy functions
        vector_JJ = self.vector_JJ
        vector_QPS = self.vector_QPS

        # Construct the full space basis change matrix for the flux-charge variables
        total_dimension = self.quadratic_hamiltonian.shape[0]
        T = self.T

        FS_basis_change_phiq = np.zeros((total_dimension, total_dimension), dtype=complex)

        FS_basis_change_phiq[:nc_total, :nc_total] = C_flux
        FS_basis_change_phiq[no_flux:nc_total + no_flux, no_flux:nc_total + no_flux] = C_charge

        FS_basis_change_phiq[nc_total:no_flux, nc_total:no_flux] = T[:T.shape[0]//2, :T.shape[1]//2]
        FS_basis_change_phiq[nc_total:no_flux, nc_total + no_flux:] = T[:T.shape[0]//2, T.shape[1]//2:]
        FS_basis_change_phiq[nc_total + no_flux:, nc_total:no_flux] = T[T.shape[0]//2:, :T.shape[1]//2]
        FS_basis_change_phiq[nc_total + no_flux:, nc_total + no_flux:] = T[T.shape[0]//2:, T.shape[1]//2:]

        # Construct the Full space almost diagonalized quadratic Hamiltonian for the flux-charge variables
        FS_quadratic_hamiltonian_phiq = np.conj(FS_basis_change_phiq.T) @ self.quadratic_hamiltonian @ FS_basis_change_phiq

        # Construct the final vectors of the JJ and QPS energy functions for the flux-charge variables
        final_vector_JJ_phiq = FS_basis_change_phiq.T @ vector_JJ
        final_vector_QPS_phiq = FS_basis_change_phiq.T @ vector_QPS


        # Construct the full space basis change matrix for the ladder operators, number-phase variables
        TG = self.T @ self.G

        FS_basis_change_an = np.zeros((total_dimension, total_dimension), dtype=complex)

        FS_basis_change_an[:nc_total, :nc_total] = C_flux
        FS_basis_change_an[no_flux:nc_total + no_flux, no_flux:nc_total + no_flux] = C_charge

        FS_basis_change_an[nc_total:no_flux, nc_total:no_flux] = TG[:TG.shape[0]//2, :TG.shape[1]//2]
        FS_basis_change_an[nc_total:no_flux, nc_total + no_flux:] = TG[:TG.shape[0]//2, TG.shape[1]//2:]
        FS_basis_change_an[nc_total + no_flux:, nc_total:no_flux] = TG[TG.shape[0]//2:, :TG.shape[1]//2]
        FS_basis_change_an[nc_total + no_flux:, nc_total + no_flux:] = TG[TG.shape[0]//2:, TG.shape[1]//2:]

        # Construct the Full space almost diagonalized quadratic Hamiltonian for the ladder operators
        FS_quadratic_hamiltonian_an = np.conj(FS_basis_change_an.T) @ self.quadratic_hamiltonian @ FS_basis_change_an

        # Construct the final vectors of the JJ and QPS energy functions (ladder operators basis)
        final_vector_JJ_an = FS_basis_change_an.T @ vector_JJ
        final_vector_QPS_an = FS_basis_change_an.T @ vector_QPS

        return (FS_quadratic_hamiltonian_phiq, FS_basis_change_phiq,
                final_vector_JJ_phiq, final_vector_QPS_phiq,
                FS_quadratic_hamiltonian_an, FS_basis_change_an,
                final_vector_JJ_an, final_vector_QPS_an)


    def diagonal_harmonic_Hamiltonian_expression(self, precision: int = 3):
        """
        Print out the diagonalized Hamiltonian. 
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
    

    def Hamiltonian_expression(self, precision: int = 3, tol: float = 1e-14):
        """
        Print out the Hamiltonian. 
        """

        # Define the matrices
        quantum_quadratic_hamiltonian = self.FS_quadratic_hamiltonian_phiq.real
        vector_JJ = self.final_vector_JJ_phiq
        vector_QPS = self.final_vector_QPS_phiq

        # Define dimensional tools
        no_flux_variables = quantum_quadratic_hamiltonian.shape[0]//2
        no_compact_fluxes = self.geom.no_final_compact_flux
        no_compact_charges = self.geom.no_final_compact_charge
        no_JJ = self.topo.no_JJ
        no_QPS = self.topo.no_QPS

        print('----------------------------------------------------------------------')

        # Print the  Hamiltonian
        print(f'Quantum Hamiltonian:')
        print(f'H/ℏ (GHz) =', end=" ")

        # Print the extended Hamiltonian (pure oscillator modes only)
        for i in range(no_compact_fluxes + no_compact_charges, no_flux_variables):
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

        # Collect JJ energies
        junction_energy = []
        for elem in self.topo.elements:
            if isinstance(elem[2], Junction):
                junction_energy.append(elem[2].value())

        for i, ej in enumerate(junction_energy):
            if i != no_JJ - 1:
                print(f' - {ej:.{precision}f} cos(v_{i+1} \u03BE\u03C6)', end=" ")
            else:
                print(f' - {ej:.{precision}f} cos(v_{i+1} \u03BE\u03C6)')

        # Collect QPS energies and print them (dual: couple to charge variables)
        phaseslip_energy = []
        for elem in self.topo.elements:
            if isinstance(elem[2], PhaseSlip):
                phaseslip_energy.append(elem[2].value())

        for i, ep in enumerate(phaseslip_energy):
            if i != no_QPS - 1:
                print(f' - {ep:.{precision}f} cos(u_{i+1} \u03BEq)', end=" ")
            else:
                print(f' - {ep:.{precision}f} cos(u_{i+1} \u03BEq)')

        if no_JJ == 0 and no_QPS == 0:
            print('')

        print('')

        np.set_printoptions(precision=precision)
        print(f'JJ coupling vectors v (flux space):')
        for i in range(vector_JJ.shape[1]):
            print(f'v_{i+1} = {(vector_JJ[:, i].real).T}')

        if no_QPS > 0:
            print('')
            print(f'QPS coupling vectors u (charge space):')
            for i in range(vector_QPS.shape[1]):
                print(f'u_{i+1} = {(vector_QPS[:, i].real).T}')

        print('')

        print(f'Flux-charge variable vector \u03BE\u03C6:')
        print(f'\u03BE\u03C6\u1D40 = (', end=" ")
        for i in range(2*no_flux_variables):
            if i < no_compact_fluxes:
                print(f'\u03D5_c{i+1}', end=" ")
            elif no_compact_fluxes <= i < no_flux_variables:
                print(f' \u03D5_e{i-no_compact_fluxes+1}', end=" ")
            elif no_flux_variables <= i < no_flux_variables + no_compact_fluxes:
                print(f' n_c{i-no_flux_variables+1}', end=" ")
            elif no_flux_variables + no_compact_fluxes <= i <= 2*no_flux_variables-1:
                print(f' n_e{i-no_compact_fluxes-no_flux_variables+1}', end=" ")
        print(f')')

        if no_QPS > 0:
            print(f'')
            print(f'Charge variable vector \u03BEq (QPS sector):')
            print(f'\u03BEq\u1D40 = (', end=" ")
            for i in range(no_compact_charges):
                print(f' q_c{i+1}', end=" ")
            for i in range(no_flux_variables - no_compact_fluxes):
                print(f' \u03C6_e{i+1}', end=" ")
            print(f')')

        print(f'')

        print(f'Operator subscripts explanation:')
        print(f' - Subindex e: extended subspace (oscillator modes)')
        print(f' - Subindex c: compact subspace (JJ flux / QPS charge)')
        print('')

        print(f'Relation between number-phase operators and flux-charge operators:')
        print(f' - n = Q/(2e)')
        print(f' - \u03D5 = 2\u03C0 \u03C6/(\u03C6_0)')
        print('----------------------------------------------------------------------')