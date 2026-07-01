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


from typing import Any

import numpy as np

from ..utils.linalg import (
    Gauge_variable_symplification,
    GaussJordan,
    integer_null_space,
    remove_zero_rows,
    reverseGaussJordan,
)
from .elements import Capacitor, Inductor, Junction, PhaseSlip

Edge = tuple[int, int, object]


class Topology:
    def __init__(self, elements_list: list[Edge], debug: bool = False) -> None:
        """
        Initializes the topology analysis.
        Processes the input list of elements to identify nodes and categorize components.

        Element ordering: [JJ | Cap | QPS | Ind]
        All elements are user-provided — no automatic companion creation.
        """
        self.debug = debug

        # ── Input validation ───────────────────────────────────────────
        if len(elements_list) == 0:
            raise ValueError("Circuit must have at least one element.")

        for a, b, elt in elements_list:
            if a == b:
                raise ValueError(
                    f"Self-loop detected: element {type(elt).__name__} "
                    f"connects node {a} to itself. "
                    "A circuit element must connect two distinct nodes."
                )

        # Check graph connectivity (union-find)
        nodes_set = set()
        for a, b, _ in elements_list:
            nodes_set.add(a)
            nodes_set.add(b)
        parent = {n: n for n in nodes_set}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for a, b, _ in elements_list:
            union(a, b)

        roots = {find(n) for n in nodes_set}
        if len(roots) > 1:
            raise ValueError(
                f"Disconnected circuit: {len(roots)} separate components detected. "
                "All nodes must be connected through circuit elements."
            )

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
        self.elements: list[list[Any]] = []
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

        # Ordering invariant: internal list must be [JJ* | Cap* | QPS* | Ind*].
        # Verified here so any future refactor of the construction loops trips
        # immediately on any circuit construction (all 379+ tests exercise this).
        _jj_end  = self.no_JJ
        _cap_end = _jj_end  + self.no_Capacitors
        _qps_end = _cap_end + self.no_QPS
        _ind_end = _qps_end + self.no_Inductors
        assert all(isinstance(e[2], Junction)  for e in self.elements[:_jj_end]),  \
            "BUG: JJ block is not at the start of self.elements"
        assert all(isinstance(e[2], Capacitor) for e in self.elements[_jj_end:_cap_end]), \
            "BUG: Capacitor block is out of order in self.elements"
        assert all(isinstance(e[2], PhaseSlip) for e in self.elements[_cap_end:_qps_end]), \
            "BUG: QPS block is out of order in self.elements"
        assert all(isinstance(e[2], Inductor)  for e in self.elements[_qps_end:_ind_end]), \
            "BUG: Inductor block is not at the end of self.elements"

        if self.debug:
            print(f"Element Counts -> JJ: {self.no_JJ}, Caps: {self.no_Capacitors}, "
                  f"QPS: {self.no_QPS}, Inds: {self.no_Inductors}, Total: {self.no_elements}")

        # Count elements per node pair for validation.
        self.cap_node_pairs = set()
        pair_counts = {}
        for a, b, elt in elements_list:
            pair = frozenset([self.node_dictionary[a], self.node_dictionary[b]])
            if pair not in pair_counts:
                pair_counts[pair] = {'JJ': 0, 'Cap': 0, 'QPS': 0, 'Ind': 0}
            if isinstance(elt, Junction):
                pair_counts[pair]['JJ'] += 1
            elif isinstance(elt, Capacitor):
                pair_counts[pair]['Cap'] += 1
                self.cap_node_pairs.add(pair)
            elif isinstance(elt, PhaseSlip):
                pair_counts[pair]['QPS'] += 1
            elif isinstance(elt, Inductor):
                pair_counts[pair]['Ind'] += 1

        # Validate element balance per node pair.
        #
        # Each element contributes ±½ to the symplectic form ω:
        #   flux-type (+½):   JJ, Ind
        #   charge-type (−½): Cap, QPS
        #
        # On the same nodes, flux-type and charge-type pair up in ω.
        # If there is an excess of nonlinear elements (JJ or QPS) beyond
        # what their linear companions (Cap or Ind) can absorb, the excess
        # cos() acts on a non-dynamical variable → nonlinear constraint
        # (Kepler-type equation) that cannot be solved.
        #
        # Rule: each nonlinear element needs a linear companion of the
        # OPPOSITE symplectic type on the same nodes:
        #   JJ (+½) needs Cap (−½)
        #   QPS (−½) needs Ind (+½)
        # One nonlinear element can be "bare" (no companion), but two or
        # more of the same type need one companion each.
        for pair, counts in pair_counts.items():
            nodes = set(pair)
            n_jj  = counts['JJ']
            n_cap = counts['Cap']
            n_qps = counts['QPS']
            n_ind = counts['Ind']

            # Cap||QPS: both charge-type (−½), they compete for flux-type
            # partners instead of pairing with each other.
            if n_cap > 0 and n_qps > 0:
                raise ValueError(
                    f"Capacitor in parallel with PhaseSlip on nodes "
                    f"{nodes} is not supported. This creates a "
                    f"nonlinear KVL constraint that cannot be fulfilled "
                    f"in general. Possible solutions: remove either the "
                    f"capacitor or the PhaseSlip, or include an inductor "
                    f"in the nonlinear capacitive loop."
                )

            # Excess QPS: N QPS need N inductors (1 bare QPS is OK).
            if n_qps > 1 and n_ind < n_qps:
                raise ValueError(
                    f"Too many PhaseSlip elements ({n_qps}) on nodes "
                    f"{nodes} with only {n_ind} inductor(s). Each QPS "
                    f"needs a companion inductor. Add {n_qps - n_ind} "
                    f"inductor(s) or remove excess PhaseSlip elements."
                )

            # Excess JJ: N JJ need N capacitors (1 bare JJ is OK).
            # Dual of excess QPS.
            if n_jj > 1 and n_cap < n_jj:
                raise ValueError(
                    f"Too many Junction elements ({n_jj}) on nodes "
                    f"{nodes} with only {n_cap} capacitor(s). Each JJ "
                    f"needs a companion capacitor. Add {n_jj - n_cap} "
                    f"capacitor(s) or remove excess Junction elements."
                )

        # Run Kirchhoff analysis
        (self.Fcut, self.Floop, self.F, self.K,
         self.no_reduced_compact_flux,
         self.no_reduced_compact_charge) = self.Kirchhoff()

    def Kirchhoff(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
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

        # Count QPS per node-pair group for compact charge mode detection.
        # For N QPS on the same node pair: max(1, N-1) compact charge modes.
        #   N=1 → 1 compact mode (the single charge).
        #   N≥2 → N-1 compact modes (the charge-difference directions).
        # This follows from KCL: for N parallel QPS, 1 scalar constraint
        # (sum of charges = const) leaves N-1 free charge differences, all compact.
        qps_start = self.no_JJ + self.no_Capacitors
        qps_group_counts: dict[frozenset[int], int] = {}
        for i in range(self.no_QPS):
            na, nb = self.elements[qps_start + i][0], self.elements[qps_start + i][1]
            pair = frozenset([na, nb])
            qps_group_counts[pair] = qps_group_counts.get(pair, 0) + 1
        no_qps_groups = len(qps_group_counts)
        # Total expected compact charge modes across all groups
        expected_compact_modes = sum(max(1, cnt - 1) for cnt in qps_group_counts.values())

        if no_one_island == 0 or no_qps_groups == 0:
            Kcut_compact = np.zeros((self.no_elements, 0))
        else:
            K_D_cut = integer_null_space(D_cut)

            # Cap at the expected number of compact charge modes.
            # Sort columns by "most Ind-zeros first": pure QPS-difference vectors
            # (zero inductor component) score higher than QPS-Ind pair vectors,
            # so the correct charge-difference modes are selected first.
            if K_D_cut.shape[1] > expected_compact_modes:
                ind_zeros = np.sum(np.abs(K_D_cut[self.no_QPS:, :]) < 1e-12, axis=0)
                order = np.argsort(-ind_zeros)  # most Ind-zeros first
                K_D_cut = K_D_cut[:, order[:expected_compact_modes]]

            # Embed into full element space (pad with zeros for two-island columns)
            Kcut_compact = np.vstack((
                np.zeros((no_two_island, K_D_cut.shape[1])),
                K_D_cut,
            ))

        no_reduced_compact_charge = Kcut_compact.shape[1]

        # Note: Cap||QPS validation is done in __init__ before Kirchhoff() runs.
        # No kcut_suppressed handling needed here.

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
                no_reduced_compact_charge)


def _independent_columns_ordered(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Return a maximal set of linearly independent columns of M,
    preserving the original column order (left-to-right priority).

    This is critical because Kloop = [compact | extended] and Kcut = [compact | extended].
    Compact columns must stay first; only redundant extended columns are dropped.
    """
    if M.shape[1] == 0:
        return M
    keep: list[int] = []
    for j in range(M.shape[1]):
        candidate = np.hstack([M[:, keep], M[:, j:j+1]]) if keep else M[:, j:j+1]
        if np.linalg.matrix_rank(candidate, tol=tol) > len(keep):
            keep.append(j)
    return M[:, keep]
