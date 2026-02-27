"""
core/topology.py

Handles graph topology analysis and Kirchhoff's laws matrix construction.
Corresponds to Step 1 (Section IIA) of the algorithm.
"""

from typing import Any, List, Tuple
import numpy as np
from scipy.linalg import null_space

from .elements import Junction, Capacitor, Inductor, PhaseSlip
from ..utils.linalg import (
    GaussJordan, 
    reverseGaussJordan, 
    remove_zero_rows, 
    GS_algorithm, 
    proportional_rows, 
    Gauge_variable_symplification
)

Edge = Tuple[int, int, object]

class Topology:
    def __init__(self, elements_list: List[Edge], debug: bool = False):   
        """
        Initializes the topology analysis. 
        Processes the input list of elements to identify nodes and categorize components.
        """
        self.debug = debug
        
        # Identify nodes
        nodes = set([a for a, _, _ in elements_list] + [b for _, b, _ in elements_list])
        self.node_dictionary = {a: i for i, a in enumerate(nodes)}
        self.no_nodes = len(self.node_dictionary)

        if self.debug:
            print("\n" + "="*50)
            print("DEBUGGING START: TOPOLOGY")
            print("="*50)
            print(f"Nodes detected: {self.node_dictionary}")

        # Categorize elements and flatten Junctions (Junction + Cap) and PhaseSlips (PhaseSlip + Ind)
        self.elements = []
        self.no_JJ = 0
        self.no_Capacitors = 0
        self.no_QPS = 0
        self.no_Inductors = 0

        # Order: [JJ | Cap (JJ parallel + regular) | PhaseSlip | Ind (QPS parallel + regular)]
        for a, b, elt in elements_list:
            if isinstance(elt, Junction):
                self.elements.append([self.node_dictionary[a], self.node_dictionary[b], elt])
                self.no_JJ += 1
        for a, b, elt in elements_list:
            if isinstance(elt, Junction):
                self.elements.append([self.node_dictionary[a], self.node_dictionary[b], elt.cap])
                self.no_Capacitors += 1
            elif isinstance(elt, Capacitor):
                self.elements.append([self.node_dictionary[a], self.node_dictionary[b], elt])
                self.no_Capacitors += 1
        for a, b, elt in elements_list:
            if isinstance(elt, PhaseSlip):
                self.elements.append([self.node_dictionary[a], self.node_dictionary[b], elt])
                self.no_QPS += 1
        for a, b, elt in elements_list:
            if isinstance(elt, PhaseSlip):
                self.elements.append([self.node_dictionary[a], self.node_dictionary[b], elt.ind])
                self.no_Inductors += 1
            elif isinstance(elt, Inductor):
                self.elements.append([self.node_dictionary[a], self.node_dictionary[b], elt])
                self.no_Inductors += 1

        self.no_elements = len(self.elements)

        if self.debug:
            print(f"Element Counts -> JJ: {self.no_JJ}, Caps: {self.no_Capacitors}, Inds: {self.no_Inductors}")

        # Run Kirchhoff analysis
        self.Fcut, self.Floop, self.F, self.K, self.no_reduced_compact_flux, self.no_reduced_compact_charge = self.Kirchhoff()

    def Kirchhoff(self):
        """
        Contructs the total Kirchhoff matrix F of the circuit and its kernel K.
        """
        if self.debug:
            print("\n" + "-"*40)
            print("1. KIRCHHOFF ANALYSIS")
            print("-"*40)
            print("Eq (1) - KCL & KVL: sum dq = 0, sum dphi = 0")
            print("Eq (2) - Constraints: F dR = 0")
            print("Eq (3) - Kernel K: K = [Kernel(F_loop), Kernel(F_cut)]")
            print("Eq (6) - Reduction: R = K Z")

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

        if self.debug:
            print(f"Fcut shape: {Fcut.shape}")
            print(f"Floop shape: {Floop.shape}")
            # print(f"Floop matrix:\n{Floop}") 

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
        if no_initial_compact_flux_variables == 0:
            # No compact flux variables (no JJ or Capacitor) → null space is trivially empty
            Kloop_S = np.zeros((self.no_elements, 0))
        else:
            Floop_S = Floop[:, :no_initial_compact_flux_variables]
            Kloop_S_raw = null_space(Floop_S)
            Kloop_S = np.vstack((Kloop_S_raw, np.zeros((self.no_elements - no_initial_compact_flux_variables, Kloop_S_raw.shape[1]))))

        no_reduced_compact_flux = Kloop_S.shape[1]  # Calculate the number of compact fluxes.

        # - Construct the full space kernel, Kloop
        Kloop_aux = Fcut.T

        if Kloop_S.shape[1] == 0:
            Kloop = Kloop_aux
        else:
            Kloop = np.block([Kloop_S, Kloop_aux])
            Kloop = GS_algorithm(Kloop, normal=True, delete_zeros=True)
        
        # - Detect and simplify compact flux variables without dynamics
        if Kloop_S.shape[1] > 1:
            proportional_rows_Kloop = proportional_rows(Kloop[:no_initial_compact_flux_variables, :])
            
            if len(proportional_rows_Kloop) > 0:

                for _, rows_group in enumerate(proportional_rows_Kloop):
                    row_idx = rows_group[0]
                    col_idx = int(np.argmax(np.abs(Kloop[row_idx, :])))
                    Kloop = Gauge_variable_symplification(Kloop, row_idx, col_idx)
        
        # ── DUAL KERNEL: compact charge (QPS) ──────────────────────────────
        # Analogous to Kloop_S for compact flux, but for compact charge (QPS).
        #
        # KEY DUALITY: just as Kloop_S includes BOTH JJ and their parallel capacitors
        # (no_initial_compact_flux = no_JJ + no_Capacitors), the dual kernel must
        # include BOTH QPS elements and their parallel inductors.
        # Reason: the compact mode emerges from the null space of the combined sector.
        #   - JJ+Cap: null_space(Floop[:, :no_JJ+no_Cap]) → loop flux mode
        #   - QPS+Ind: null_space(Fcut[:, qps_start:qps_start+2*no_QPS]) → loop charge mode
        no_initial_compact_charge_variables = 2 * self.no_QPS   # QPS + their parallel inductors

        # Compact-charge columns of Fcut: [QPS cols | QPS-inductor cols]
        qps_start = self.no_JJ + self.no_Capacitors
        Fcut_S = Fcut[:, qps_start : qps_start + no_initial_compact_charge_variables]

        if no_initial_compact_charge_variables > 0:
            Kcut_S = null_space(Fcut_S)
            # Pad to full element count (no_elements rows), zeroing non-QPS/Ind rows
            Kcut_S_full = np.zeros((self.no_elements, Kcut_S.shape[1]))
            Kcut_S_full[qps_start : qps_start + no_initial_compact_charge_variables, :] = Kcut_S
        else:
            Kcut_S_full = np.zeros((self.no_elements, 0))

        no_reduced_compact_charge = Kcut_S_full.shape[1]

        # Construct full Kcut (dual of Kloop construction)
        Kcut_aux = Floop.T
        if Kcut_S_full.shape[1] == 0:
            Kcut = Kcut_aux
        else:
            Kcut = np.block([Kcut_S_full, Kcut_aux])
            Kcut = GS_algorithm(Kcut, normal=True, delete_zeros=True)

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

        if self.debug:
            print(f"Total F shape: {F.shape}, Rank: {np.linalg.matrix_rank(F)}")
            print(f"Total K shape: {K.shape}, Rank: {np.linalg.matrix_rank(K)}")
            check = np.max(np.abs(F @ K))
            print(f"Verification F @ K ~ 0: {check:.2e}")
            print(f"Compact charge variables (QPS): {no_reduced_compact_charge}")

        return Fcut, Floop, F, K, no_reduced_compact_flux, no_reduced_compact_charge