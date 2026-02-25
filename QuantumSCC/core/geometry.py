"""
core/geometry.py

Handles the symplectic form construction and Faddeev-Jackiw reduction.
Corresponds to Step 2 (Section IIB) of the algorithm.
"""

import numpy as np
from .elements import Junction, Capacitor, Inductor, PhaseSlip
from ..utils.linalg import omega_symplectic_transformation

class Geometry:
    def __init__(self, topology, debug: bool = False):
        """
        Initializes the geometry analysis using the results from the Topology step.
        """
        self.debug = debug    
        self.topo = topology
        
        # Run symplectic analysis
        self.omega_2B, self.omega_symplectic, self.V, \
        self.no_independent_variables, self.no_final_compact_flux, \
        self.no_final_compact_charge = self.omega_function()

    def omega_function(self):
        """
        Given the Lagrangian of the circuit: Lagrangian = omega - energy. It calculates the symplectic form of 
        the two-form omega and the basis change matrix.
        """
        if self.debug:
            print("\n" + "-"*40)
            print("2. SYMPLECTIC GEOMETRY")
            print("-"*40)
            print("Eq (8) - Symplectic Form: omega_2B = 1/2 dR^T ^ Omega_2B dR")
            print("Eq (14) - Canonical Form: V^T Omega V = J")
            print("Eq (16) - Basis Change: Z = V (xi, w)^T")

        # Obtain omega_2B matrix
        omega_2B = np.zeros((2 * self.topo.no_elements, 2 * self.topo.no_elements))
        for i, elem in enumerate(self.topo.elements):

            if isinstance(elem[2], Junction):
                omega_2B[i, i + self.topo.no_elements] = 0.5
                omega_2B[i + self.topo.no_elements, i] = -0.5

            elif isinstance(elem[2], Capacitor):
                omega_2B[i, i + self.topo.no_elements] = -0.5
                omega_2B[i + self.topo.no_elements, i] = 0.5

            elif isinstance(elem[2], PhaseSlip):
                # QPS has compact charge q_P ∈ S¹ → same sign convention as Capacitor
                omega_2B[i, i + self.topo.no_elements] = -0.5
                omega_2B[i + self.topo.no_elements, i] = 0.5

            elif isinstance(elem[2], Inductor):
                omega_2B[i, i + self.topo.no_elements] = 0.5
                omega_2B[i + self.topo.no_elements, i] = -0.5

        if self.debug:
            print(f"Omega 2B shape: {omega_2B.shape}")
            print(f"Determinant of Omega 2B: {np.linalg.det(omega_2B):.6f}")

        # Obtain omega matrix in the Kirchhoff equations basis
        omega_non_symplectic = self.topo.K.T @ omega_2B @ self.topo.K

        # Obtain the symplectic form of the omega matrix and the basis change matrix
        omega_symplectic, V, no_final_compact_flux, no_final_compact_charge = omega_symplectic_transformation(
            omega_non_symplectic,
            no_compact_flux_variables=self.topo.no_reduced_compact_flux,
            no_compact_charge_variables=self.topo.no_reduced_compact_charge,
            no_flux_variables=self.topo.Fcut.shape[0]
        )

        # Remove the zeros columns and rows from omega_symplectic
        no_independent_variables = np.linalg.matrix_rank(omega_symplectic)
        omega_symplectic = omega_symplectic[:no_independent_variables, :no_independent_variables]

        assert no_final_compact_flux <= no_independent_variables//2, \
            "There is an error, the number of compact fluxes must be equal or smaller than the number of total fluxes"

        if self.debug:
            print(f"Omega Symplectic shape: {omega_symplectic.shape}")
            print(f"Basis Change V shape: {V.shape}")
            print(f"Independent Variables: {no_independent_variables}")
            print(f"Final compact flux variables: {no_final_compact_flux}")
            print(f"Final compact charge variables: {no_final_compact_charge}")

        return omega_2B, omega_symplectic, V, no_independent_variables, no_final_compact_flux, no_final_compact_charge