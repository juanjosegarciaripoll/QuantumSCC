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

        # Collect node pairs of standalone Capacitor elements (not JJ-parallel caps).
        # Used in Kirchhoff() to detect QPS elements shunted by a bare capacitor,
        # which suppresses the compact charge mode (dual of a shunting inductor
        # suppressing the compact flux mode in fluxonium).
        self.bare_cap_node_pairs = set()
        for a, b, elt in elements_list:
            if isinstance(elt, Capacitor):
                pair = frozenset([self.node_dictionary[a], self.node_dictionary[b]])
                self.bare_cap_node_pairs.add(pair)

        # Collect node pairs with at least one JJ.
        # Used in quantization.py to detect JJ ∥ QPS overlap (crossed pairing).
        self.jj_node_pairs = set()
        for a, b, elt in elements_list:
            if isinstance(elt, Junction):
                self.jj_node_pairs.add(frozenset([self.node_dictionary[a], self.node_dictionary[b]]))

        # Run Kirchhoff analysis
        self.Fcut, self.Floop, self.F, self.K, self.no_reduced_compact_flux, self.no_reduced_compact_charge, self.kcut_suppressed, self.qps_groups = self.Kirchhoff()

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
        # For each unique QPS node pair, exactly one compact charge mode exists.
        # Multiple QPS on the same node pair share one mode — just as multiple
        # JJ in parallel share one compact flux mode.
        #
        # STRUCTURAL ASYMMETRY vs JJ case:
        #   JJ: Floop_S has (2N-1) rows for N JJ on 2 nodes → null_space gives 1 vector ✓
        #   QPS: Fcut_S always has (no_nodes-1) rows regardless of N → for 2 nodes,
        #        null_space of 1×2N matrix gives 2N-1 vectors (overcounts by N-1).
        #
        # FIX: compute null_space from ONE representative (QPS, Ind) pair per unique
        # node pair, then broadcast that compact mode to all parallel QPS+Ind pairs.
        # This yields exactly one compact charge mode per unique QPS node pair.
        qps_start = self.no_JJ + self.no_Capacitors

        if self.no_QPS == 0:
            Kcut_S_full = np.zeros((self.no_elements, 0))
            no_reduced_compact_charge = 0
            qps_groups = {}
        else:
            # Group QPS indices (within 0..no_QPS-1) by their node pair
            qps_groups = {}
            for i in range(self.no_QPS):
                na, nb = self.elements[qps_start + i][0], self.elements[qps_start + i][1]
                pair = frozenset([na, nb])
                if pair not in qps_groups:
                    qps_groups[pair] = []
                qps_groups[pair].append(i)

            # One compact charge mode per unique node pair
            Kcut_S_full = np.zeros((self.no_elements, len(qps_groups)))
            for col_idx, (pair, qps_indices) in enumerate(qps_groups.items()):
                # Representative: first QPS in this group + its parallel inductor
                rep = qps_indices[0]
                rep_qps_col = qps_start + rep
                rep_ind_col = qps_start + self.no_QPS + rep

                # Compact charge mode from representative pair (1D null space of a
                # (no_nodes-1)×2 matrix, guaranteed 1D when QPS and Ind share nodes)
                ns = null_space(Fcut[:, [rep_qps_col, rep_ind_col]])
                v_qps, v_ind = ns[0, 0], ns[1, 0]

                # Apply the same compact mode to ALL (QPS, Ind) pairs in this group
                for i in qps_indices:
                    Kcut_S_full[qps_start + i, col_idx] = v_qps
                    Kcut_S_full[qps_start + self.no_QPS + i, col_idx] = v_ind

            no_reduced_compact_charge = len(qps_groups)

        # Suppress compact charge modes for QPS node pairs shunted by a bare capacitor.
        # A bare capacitor in parallel with a QPS makes the QPS charge extended
        # (exactly dual to a parallel inductor making JJ flux extended in fluxonium).
        # Suppression is counted per unique node pair, not per individual QPS element.
        kcut_suppressed = False
        if no_reduced_compact_charge > 0:
            suppressed_pairs = set(
                frozenset([self.elements[qps_start + i][0], self.elements[qps_start + i][1]])
                for i in range(self.no_QPS)
                if frozenset([self.elements[qps_start + i][0], self.elements[qps_start + i][1]])
                in self.bare_cap_node_pairs
            )
            n_suppressed = len(suppressed_pairs)
            n_total_groups = len(qps_groups)

            if n_suppressed == n_total_groups:
                no_reduced_compact_charge = 0
                Kcut_S_full = np.zeros((self.no_elements, 0))
                kcut_suppressed = True
            elif n_suppressed > 0:
                raise NotImplementedError(
                    f"{n_suppressed} of {n_total_groups} QPS node pairs are suppressed "
                    "by a parallel capacitor. Partial suppression is not yet supported."
                )

        # Construct full Kcut (dual of Kloop construction).
        # When compact charge modes were suppressed by a parallel capacitor, apply
        # GS_algorithm to Kcut_aux = Floop.T so that QPS charge vectors spread
        # correctly across multiple K-space directions rather than aligning entirely
        # with a single (gauge) basis vector.  For circuits without suppression,
        # the existing behaviour (Kcut = Kcut_aux unmodified) is preserved to avoid
        # altering canonical variable choices in validated test cases.
        Kcut_aux = Floop.T
        if Kcut_S_full.shape[1] == 0:
            if kcut_suppressed:
                Kcut = GS_algorithm(Kcut_aux, normal=True, delete_zeros=True)
            else:
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

        return Fcut, Floop, F, K, no_reduced_compact_flux, no_reduced_compact_charge, kcut_suppressed, qps_groups