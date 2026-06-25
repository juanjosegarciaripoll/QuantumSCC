"""
circuit.py contains the classes for the circuit and their properties.
Refactored to use the core modules.
"""

from typing import Any, Tuple, List

from .core.elements import Capacitor, Inductor, Junction, PhaseSlip
from .core.topology import Topology
from .core.geometry import Geometry
from .core.quantization import Quantization

Edge = Tuple[int, int, object]

class Circuit:
    """
    Class that contains circuit properties.
    Orchestrates the Topological, Geometrical, and Quantization analyses.

    Parameters
    ----------
        elements:
            A list of tuples that contains the circuit's elements at each branch.
        debug:
            Boolean flag to enable detailed print output of the derivation steps.
    """

    def __init__(self, elements: List[Edge], debug: bool = False) -> None:
        """Define a circuit from a list of edges and circuit elements."""
        
        self.debug = debug
        if self.debug:
            print("\n" + "="*50)
            print("INITIALIZING CIRCUIT ANALYSIS (Debug Mode ON)")
            print("="*50)

        # 1. Topological Analysis (Kirchhoff)
        self.topo = Topology(elements, debug=self.debug)
        
        # Expose topological attributes for backward compatibility / inspection
        self.elements = self.topo.elements
        self.no_elements = self.topo.no_elements
        self.no_nodes = self.topo.no_nodes
        self.node_dict = self.topo.node_dictionary
        self.no_JJ = self.topo.no_JJ
        self.no_Capacitors = self.topo.no_Capacitors
        self.no_QPS = self.topo.no_QPS
        self.no_Inductors = self.topo.no_Inductors

        self.Fcut = self.topo.Fcut
        self.Floop = self.topo.Floop
        self.F = self.topo.F
        self.K = self.topo.K
        self.no_reduced_compact_flux = self.topo.no_reduced_compact_flux
        self.no_reduced_compact_charge = self.topo.no_reduced_compact_charge

        # 2. Geometrical Analysis (Symplectic Form)
        self.geom = Geometry(self.topo, debug=self.debug)
        
        # Expose geometrical attributes
        self.omega_2B = self.geom.omega_2B
        self.omega_symplectic = self.geom.omega_symplectic
        self.V = self.geom.V
        self.no_independent_variables = self.geom.no_independent_variables
        self.no_final_compact_flux = self.geom.no_final_compact_flux
        self.no_final_compact_charge = self.geom.no_final_compact_charge

        # 3. Quantization Analysis (Hamiltonian)
        self.quant = Quantization(self.topo, self.geom, debug=self.debug)
        
        # Expose quantization attributes
        self.quadratic_hamiltonian = self.quant.quadratic_hamiltonian
        self.vector_JJ = self.quant.vector_JJ
        self.vector_QPS = self.quant.vector_QPS
        self.extended_quantum_hamiltonian = self.quant.extended_quantum_hamiltonian
        self.T = self.quant.T
        self.G = self.quant.G

        # Nonlinear attributes
        self.FS_quadratic_hamiltonian_phiq = self.quant.FS_quadratic_hamiltonian_phiq
        self.FS_basis_change_phiq = self.quant.FS_basis_change_phiq
        self.final_vector_JJ_phiq = self.quant.final_vector_JJ_phiq
        self.final_vector_QPS_phiq = self.quant.final_vector_QPS_phiq
        self.FS_quadratic_hamiltonian_an = self.quant.FS_quadratic_hamiltonian_an
        self.FS_basis_change_an = self.quant.FS_basis_change_an
        self.final_vector_JJ_an = self.quant.final_vector_JJ_an
        self.final_vector_QPS_an = self.quant.final_vector_QPS_an

        if self.debug:
            print("\n" + "="*50)
            print("END OF DEBUGGING")
            print("="*50 + "\n")

    def Kirchhoff(self):
        """Wrapper for consistency checking, if needed manually."""
        return self.topo.Kirchhoff()
    
    def omega_function(self):
        """Wrapper for consistency checking, if needed manually."""
        return self.geom.omega_function()
        
    def classical_hamiltonian_function(self):
        return self.quant.classical_hamiltonian_function()

    def extended_hamiltonian_quantization(self):
        return self.quant.extended_hamiltonian_quantization()

    def total_hamiltonian_quantization(self):
        return self.quant.total_hamiltonian_quantization()

    def diagonal_harmonic_Hamiltonian_expression(self, precision: int = 3):
        """[Terminal] Print harmonic frequencies in diagonal normal-mode basis."""
        self.quant.diagonal_harmonic_Hamiltonian_expression(precision)

    def Hamiltonian_expression(self, precision: int = 3, tol: float = 1e-14, verbose: bool = True):
        """[Terminal] Print the numerical Hamiltonian.
        verbose=True (default): full output with coupling vectors, variable legend, operator explanations.
        verbose=False: only the H/ℏ expression line.
        Works in both terminal scripts and Jupyter notebooks."""
        self.quant.Hamiltonian_expression(precision, tol, verbose)

    def symbolic_hamiltonian_expression(self, precision: int = 3, tol: float = 1e-9, verbose: bool = True):
        """[Terminal + Jupyter] Print the Hamiltonian with symbolic energy parameters
        (E_C, E_L, E_J, E_P) in reduced Darboux variables.
        verbose=True (default): adds variable legend, parameter values, and full numerical Hamiltonian.
        verbose=False: only the H/ℏ = ... symbolic expression line.

        - In Jupyter: renders LaTeX via IPython.display.Math.
        - In terminal: single-line Unicode format.
        """
        self.quant.symbolic_hamiltonian_expression(precision, tol, verbose)