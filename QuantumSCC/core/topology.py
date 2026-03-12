"""
core/topology.py

Handles graph topology analysis and Kirchhoff's laws matrix construction.
Implements the two-topology procedure from PRX 2025 Eq. 42-44:

  Flux topology (KVL):   F_loop = [D_loop | E_loop]
  Charge topology (KCL): F_cut  = [E_cut  | D_cut ]

  D_loop = Floop[:, two-island]  (JJ + Cap)  →  ker(D_loop) = compact flux
  E_loop = Floop[:, one-island]  (QPS + Ind) →  extended flux
  E_cut  = Fcut[:, two-island]   (JJ + Cap)  →  extended charge
  D_cut  = Fcut[:, one-island]   (QPS + Ind) →  ker(D_cut)  = compact charge
"""

from typing import List, Tuple
import numpy as np

from .elements import Junction, Capacitor, Inductor, PhaseSlip
from ..utils.linalg import (
    GaussJordan,
    reverseGaussJordan,
    remove_zero_rows,
    integer_null_space,
    Gauge_variable_symplification,
)

Edge = Tuple[int, int, object]


class Topology:
    def __init__(self, elements_list: List[Edge], debug: bool = False):
        """
        Initializes the topology analysis.
        Processes the input list of elements to identify nodes and categorize components.

        Element ordering: [JJ | Cap | QPS | Ind]
        All elements are user-provided — no automatic companion creation.
        """
        self.debug = debug

        # Identify nodes
        nodes = set([a for a, _, _ in elements_list] + [b for _, b, _ in elements_list])
        self.node_dictionary = {a: i for i, a in enumerate(nodes)}
        self.no_nodes = len(self.node_dictionary)

        if self.debug:
            print("\n" + "=" * 50)
            print("DEBUGGING START: TOPOLOGY")
            print("=" * 50)
            print(f"Nodes detected: {self.node_dictionary}")

        # Categorize elements — order: [JJ | Cap | QPS | Ind]
        # All elements are user-provided. No automatic companion creation.
        self.elements = []
        self.no_JJ = 0
        self.no_Capacitors = 0
        self.no_QPS = 0
        self.no_Inductors = 0

        # JJ branches
        for a, b, elt in elements_list:
            if isinstance(elt, Junction):
                self.elements.append([self.node_dictionary[a], self.node_dictionary[b], elt])
                self.no_JJ += 1

        # Capacitor branches (all user-provided)
        for a, b, elt in elements_list:
            if isinstance(elt, Capacitor):
                self.elements.append([self.node_dictionary[a], self.node_dictionary[b], elt])
                self.no_Capacitors += 1

        # QPS branches
        for a, b, elt in elements_list:
            if isinstance(elt, PhaseSlip):
                self.elements.append([self.node_dictionary[a], self.node_dictionary[b], elt])
                self.no_QPS += 1

        # Inductor branches (all user-provided)
        for a, b, elt in elements_list:
            if isinstance(elt, Inductor):
                self.elements.append([self.node_dictionary[a], self.node_dictionary[b], elt])
                self.no_Inductors += 1

        self.no_elements = len(self.elements)

        if self.debug:
            print(f"Element Counts -> JJ: {self.no_JJ}, Caps: {self.no_Capacitors}, "
                  f"QPS: {self.no_QPS}, Inds: {self.no_Inductors}, Total: {self.no_elements}")

        # Collect node pairs of Capacitor elements.
        # Used to detect QPS shunted by a capacitor → suppresses compact charge.
        self.cap_node_pairs = set()
        for a, b, elt in elements_list:
            if isinstance(elt, Capacitor):
                pair = frozenset([self.node_dictionary[a], self.node_dictionary[b]])
                self.cap_node_pairs.add(pair)

        # Run Kirchhoff analysis
        (self.Fcut, self.Floop, self.F, self.K,
         self.no_reduced_compact_flux, self.no_reduced_compact_charge,
         self.kcut_suppressed) = self.Kirchhoff()

    def Kirchhoff(self):
        """
        Constructs the total Kirchhoff matrix F and its kernel K.

        Follows PRX 2025 Eq. 42-44, two-topology decomposition:

          Flux topology (KVL):   F_loop dφ = D_loop dθ + E_loop dz = 0
          Charge topology (KCL): F_cut  dq = E_cut dz_Q + D_cut dθ_Q = 0

        Column partition [two-island | one-island] = [JJ + Cap | QPS + Ind]:
          D_loop = Floop[:, two-island]  →  ker(D_loop) = compact flux
          E_loop = Floop[:, one-island]  →  extended flux
          E_cut  = Fcut[:, two-island]   →  extended charge
          D_cut  = Fcut[:, one-island]   →  ker(D_cut)  = compact charge

        Stores D_loop, E_loop, D_cut, E_cut as attributes for article fidelity.
        """
        if self.debug:
            print("\n" + "-" * 40)
            print("1. KIRCHHOFF ANALYSIS (Eq. 42)")
            print("-" * 40)

        # ── Build Fcut (KCL) and Floop (KVL) ────────────────────────────
        Fcut_raw = np.zeros((self.no_nodes, self.no_elements))
        for n_edge, (orig_node, dest_node, _) in enumerate(self.elements):
            Fcut_raw[orig_node, n_edge] = -1
            Fcut_raw[dest_node, n_edge] = +1

        Fcut_raw, order = GaussJordan(Fcut_raw)
        Fcut = reverseGaussJordan(remove_zero_rows(Fcut_raw))

        n = len(Fcut)
        A = Fcut[:, n:]
        Floop = np.hstack((-A.T, np.eye(A.shape[1])))

        # Restore original column ordering
        inv_order = np.argsort(order)
        Fcut = Fcut[:, inv_order]
        Floop = Floop[:, inv_order]

        if self.debug:
            print(f"Fcut shape: {Fcut.shape}")
            print(f"Floop shape: {Floop.shape}")

        # ── Full Kirchhoff matrix F = [[Floop, 0], [0, Fcut]] ───────────
        F = np.block([
            [Floop, np.zeros((Floop.shape[0], Fcut.shape[1]))],
            [np.zeros((Fcut.shape[0], Floop.shape[1])), Fcut],
        ])

        # ── Column classification (Eq. 42) ───────────────────────────────
        # Two-island elements (JJ + Cap): compact flux candidates (S¹ topology)
        # One-island elements (QPS + Ind): compact charge candidates
        no_two_island = self.no_JJ + self.no_Capacitors
        no_one_island = self.no_QPS + self.no_Inductors

        # ── PART 1: Compact flux — Eq. 42 flux topology ──────────────────
        # F_loop = [D_loop | E_loop]
        #   D_loop = Floop[:, two-island]  → ker(D_loop) = compact flux
        #   E_loop = Floop[:, one-island]  → extended flux directions
        D_loop = Floop[:, :no_two_island]
        E_loop = Floop[:, no_two_island:]

        if no_two_island == 0:
            Kloop_compact = np.zeros((self.no_elements, 0))
        else:
            K_D_loop = integer_null_space(D_loop)
            # Embed into full element space (pad with zeros for one-island columns)
            Kloop_compact = np.vstack((
                K_D_loop,
                np.zeros((no_one_island, K_D_loop.shape[1]))
            ))

        no_reduced_compact_flux = Kloop_compact.shape[1]

        # Extended flux directions: orthogonal complement of compact within ker(Floop)
        Kloop_extended = Fcut.T

        # Combine: [compact | extended], then clean up linear dependence
        if Kloop_compact.shape[1] == 0:
            Kloop = Kloop_extended
        else:
            Kloop = np.hstack([Kloop_compact, Kloop_extended])
            Kloop = _independent_columns_ordered(Kloop)

        # Detect and simplify gauge (non-dynamical) compact flux variables
        if no_reduced_compact_flux > 1:
            from ..utils.linalg import proportional_rows
            prop_groups = proportional_rows(Kloop[:no_two_island, :])
            for rows_group in prop_groups:
                row_idx = rows_group[0]
                col_idx = int(np.argmax(np.abs(Kloop[row_idx, :])))
                Kloop = Gauge_variable_symplification(Kloop, row_idx, col_idx)

        if self.debug:
            print(f"  D_loop shape: {D_loop.shape}  (Floop restricted to JJ+Cap)")
            print(f"  E_loop shape: {E_loop.shape}  (Floop restricted to QPS+Ind)")
            print(f"Compact flux variables: {no_reduced_compact_flux}")
            print(f"Kloop shape: {Kloop.shape}")

        # ── PART 2: Compact charge — Eq. 42 charge topology ─────────────
        # F_cut = [E_cut | D_cut]
        #   E_cut = Fcut[:, two-island]  → extended charge
        #   D_cut = Fcut[:, one-island]  → ker(D_cut) = compact charge
        #
        # Symmetric with compact flux detection: just compute ker(D_cut) directly.
        E_cut = Fcut[:, :no_two_island]
        D_cut = Fcut[:, no_two_island:]

        # Count distinct QPS node pairs for compact charge cap.
        qps_start = self.no_JJ + self.no_Capacitors
        qps_node_pairs = set()
        for i in range(self.no_QPS):
            na, nb = self.elements[qps_start + i][0], self.elements[qps_start + i][1]
            qps_node_pairs.add(frozenset([na, nb]))
        no_qps_groups = len(qps_node_pairs)

        if no_one_island == 0 or no_qps_groups == 0:
            Kcut_compact = np.zeros((self.no_elements, 0))
        else:
            K_D_cut = integer_null_space(D_cut)

            # Cap at number of distinct QPS node-pair groups.
            # The D_cut kernel can be larger than the physical compact charge
            # count when multiple QPS+Ind share the same nodes.  Each QPS
            # node-pair group contributes at most 1 compact charge mode.
            # Sort columns to prefer QPS-pure directions (zero Ind components)
            # so the first columns are the best compact representatives.
            if K_D_cut.shape[1] > no_qps_groups:
                ind_zeros = np.sum(np.abs(K_D_cut[self.no_QPS:, :]) < 1e-12, axis=0)
                order = np.argsort(-ind_zeros)  # most Ind-zeros first
                K_D_cut = K_D_cut[:, order[:no_qps_groups]]

            # Embed into full element space (pad with zeros for two-island columns)
            Kcut_compact = np.vstack((
                np.zeros((no_two_island, K_D_cut.shape[1])),
                K_D_cut,
            ))

        no_reduced_compact_charge = Kcut_compact.shape[1]

        # Suppress compact charge for QPS shunted by a capacitor on the same nodes.
        # A parallel capacitor provides a charge shunt path, making QPS charge extended.
        kcut_suppressed = False
        if no_reduced_compact_charge > 0 and no_qps_groups > 0:
            suppressed_pairs = {p for p in qps_node_pairs if p in self.cap_node_pairs}
            n_suppressed = len(suppressed_pairs)
            n_total = no_qps_groups

            if n_suppressed == n_total:
                no_reduced_compact_charge = 0
                Kcut_compact = np.zeros((self.no_elements, 0))
                kcut_suppressed = True
            elif n_suppressed > 0:
                raise NotImplementedError(
                    f"{n_suppressed} of {n_total} QPS node pairs are suppressed "
                    "by a parallel capacitor. Partial suppression is not yet supported."
                )

        # Extended charge directions: from Floop.T
        Kcut_extended = Floop.T

        if Kcut_compact.shape[1] == 0:
            Kcut = Kcut_extended
        else:
            Kcut = np.hstack([Kcut_compact, Kcut_extended])
            Kcut = _independent_columns_ordered(Kcut)

        if self.debug:
            print(f"  E_cut shape: {E_cut.shape}  (Fcut restricted to JJ+Cap)")
            print(f"  D_cut shape: {D_cut.shape}  (Fcut restricted to QPS+Ind)")
            print(f"Compact charge variables: {no_reduced_compact_charge}")
            print(f"Kcut shape: {Kcut.shape}")

        # ── Build total kernel K = block_diag(Kloop, Kcut) ──────────────
        K = np.block([
            [Kloop, np.zeros((Kloop.shape[0], Kcut.shape[1]))],
            [np.zeros((Kcut.shape[0], Kloop.shape[1])), Kcut],
        ])

        # Verify
        assert np.allclose(F @ K, 0), "Error: F @ K != 0"
        expected_dim = F.shape[1] - np.linalg.matrix_rank(F)
        assert K.shape[1] == expected_dim, \
            f"Kernel dimension mismatch: K has {K.shape[1]} cols, expected {expected_dim}"

        if self.debug:
            print(f"Total F shape: {F.shape}, Rank: {np.linalg.matrix_rank(F)}")
            print(f"Total K shape: {K.shape}")
            print(f"Verification F @ K ~ 0: {np.max(np.abs(F @ K)):.2e}")

        # ── Store Eq. 42 sub-matrices ─────────────────────────────────────
        self.D_loop = D_loop   # Floop[:, two-island]  — ker → compact flux
        self.E_loop = E_loop   # Floop[:, one-island]  — extended flux
        self.E_cut  = E_cut    # Fcut[:, two-island]   — extended charge
        self.D_cut  = D_cut    # Fcut[:, one-island]   — ker → compact charge

        return (Fcut, Floop, F, K, no_reduced_compact_flux,
                no_reduced_compact_charge, kcut_suppressed)


def _independent_columns_ordered(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Return a maximal set of linearly independent columns of M,
    preserving the original column order (left-to-right priority).

    This is critical because Kloop = [compact | extended] and Kcut = [compact | extended].
    Compact columns must stay first; only redundant extended columns are dropped.
    """
    if M.shape[1] == 0:
        return M
    keep = []
    for j in range(M.shape[1]):
        candidate = np.hstack([M[:, keep], M[:, j:j+1]]) if keep else M[:, j:j+1]
        if np.linalg.matrix_rank(candidate, tol=tol) > len(keep):
            keep.append(j)
    return M[:, keep]
