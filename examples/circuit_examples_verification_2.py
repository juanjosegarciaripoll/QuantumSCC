"""
Verification examples — Part 2: Dualmon variants and 4-node JJ+QPS networks.

Explores:
  - Bare dualmon and dualmon with inductors/capacitors
  - Cap||QPS error detection (Kepler equation)
  - Dualmon with series/parallel L and C combinations
  - 4-node network with JJ and QPS
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: F401 — used by callers of show()

from QuantumSCC import Capacitor, Circuit, Inductor, Junction, PhaseSlip

SEP = "\n" + "=" * 70


def show(name, edges, explain=""):
    """Build circuit, print topology, symbolic and numerical H."""
    print(SEP)
    print(name)
    if explain:
        print(explain)
    print()

    try:
        c = Circuit(edges)
    except Exception as e:
        print(f"  ERROR: {e}")
        print()
        return None

    topo = c.topo
    nCF = c.geom.no_final_compact_flux
    nCC = c.geom.no_final_compact_charge
    nF = c.geom.no_independent_variables // 2
    nEF = nF - nCF - nCC

    print("Elements:")
    for i, e in enumerate(topo.elements):
        val = e[2].energy() if hasattr(e[2], 'energy') else e[2].value()
        print(f"  {i}: {type(e[2]).__name__:12s} ({e[0]},{e[1]})  E={val:.2f} GHz")

    nodes = sorted(set(a for a, b, _ in topo.elements) | set(b for a, b, _ in topo.elements))
    print(f"Nodes: {nodes} (0=ground)")
    print(f"Modes: nCF={nCF}, nCC={nCC}, nEF={nEF} -> {nF} pairs")

    # List conjugate pairs in braket order using Circuit method
    pairs = c.conjugate_pairs()
    type_labels = {
        'JJ_compact': 'JJ compact flux, integer charge',
        'QPS_compact': 'QPS-inductor flux, compact charge',
        'extended': 'extended oscillator mode',
    }
    print("Conjugate pairs (braket order):")
    for i, (flux, charge, ptype) in enumerate(pairs):
        print(f"  pair {i+1}: ({flux}, {charge})   [{type_labels[ptype]}]")

    print()

    c.symbolic_hamiltonian_expression(verbose=False)
    c.Hamiltonian_expression(verbose=False)
    return c


# =====================================================================
# SECTION 1: Dualmon variants
# =====================================================================

print(SEP)
print("SECTION 1: DUALMON VARIANTS")
print("=" * 70)
print("""
The basic dualmon is JJ(0,1) + QPS(0,1): two nonlinear elements,
no quadratic energy, H = -E_J cos(phi) - E_P cos(q).

Adding a Capacitor on the SAME nodes as a QPS is NOT allowed:
both are charge-type (-1/2), and their contributions cancel in omega,
making the QPS charge a gauge variable. The constraint dH/dw = 0
becomes a Kepler equation (nonlinear, unsolvable).

An Inductor on the same nodes as JJ+QPS provides dynamics and
keeps the QPS compact charge alive.
""")

# --- 1a: Bare dualmon (no dynamics) ---
show("1a: Bare dualmon -- JJ + QPS (no quadratic energy)",
     [(0, 1, Junction(5.0)),
      (0, 1, PhaseSlip(3.0))],
     """Topology (2 nodes: 0=ground, 1):
  JJ  (0,1) -- Josephson junction, E_J=5 GHz
  QPS (0,1) -- phase-slip element, E_P=3 GHz

No C, no L -> no quadratic H. Only cos(phi) + cos(q).
Both variables are compact: phi in S^1, q in S^1.""")

# --- 1b: Dualmon + C on same nodes as QPS (ERROR: Kepler) ---
show("1b: Dualmon + C on QPS nodes (0,1) -- ERROR: Cap||QPS",
     [(0, 1, Junction(5.0)),
      (0, 1, PhaseSlip(3.0)),
      (0, 1, Capacitor(1.0))],
     """Topology (2 nodes: 0=ground, 1):
  JJ  (0,1), QPS (0,1), C (0,1)

Cap and QPS are both charge-type (-1/2). On the same nodes, their
contributions to omega cancel, making the QPS charge a gauge variable.
The QPS cos(q) then depends on a gauge variable with quadratic energy
from the capacitor -> Kepler equation -> unsolvable.
Expected: ValueError (Capacitor in parallel with PhaseSlip).""")

# --- 1c: Dualmon + L on same nodes (gives oscillator) ---
show("1c: Dualmon + L on (0,1) -- inductor gives dynamics",
     [(0, 1, Junction(5.0)),
      (0, 1, PhaseSlip(3.0)),
      (0, 1, Inductor(1.0))],
     """Topology (2 nodes: 0=ground, 1):
  JJ  (0,1), QPS (0,1), L (0,1)

L (+1/2) pairs with QPS (-1/2) -> dynamics for compact charge.
L in JJ loop -> kills JJ compact flux (extends it to R).
Result: nCF=0, nCC=1.""")

# --- 1d: Full dualmon (JJ+C on node pair, L series, QPS on other pair) ---
show("1d: Full dualmon -- JJ+C(1,0), L(1,2), QPS(2,0)",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  JJ  (1,0) + C (1,0) -- transmon-like on nodes (1,0)
  L   (1,2) -- inductor connecting JJ side to QPS side
  QPS (2,0) -- phase-slip on nodes (2,0)

Classic dualmon from Le et al. (arXiv:1904.01843).
C is on JJ nodes (1,0), NOT on QPS nodes (2,0) -> no Cap||QPS conflict.
L in JJ loop (0->1->2->0) -> kills JJ compact flux.""")

# --- 1e: Full dualmon + C on QPS nodes (ERROR: Cap||QPS) ---
show("1e: Full dualmon + C on QPS nodes (2,0) -- ERROR: Cap||QPS",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0)),
      (2, 0, Capacitor(2.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  JJ (1,0), C1 (1,0), L (1,2), QPS (2,0), C2 (2,0)

C2 on nodes (2,0) = same as QPS -> Cap||QPS conflict.
Expected: ValueError (Capacitor in parallel with PhaseSlip).""")

# --- 1f: Full dualmon + C on DIFFERENT nodes (valid) ---
show("1f: Full dualmon + C on different nodes (1,0) -- valid",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0)),
      (1, 0, Capacitor(2.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  JJ (1,0), C1 (1,0), C2 (1,0), L (1,2), QPS (2,0)

C2 on (1,0) = JJ nodes, NOT QPS nodes -> no Cap||QPS conflict.
Parallel caps on JJ nodes: C_total = C1 + C2.""")


# =====================================================================
# SECTION 2: Dualmon with series/parallel L and C combinations
# =====================================================================

print()
print(SEP)
print("SECTION 2: DUALMON WITH SERIES/PARALLEL COMBINATIONS")
print("=" * 70)

# --- 2a: Dualmon + 2 inductors in parallel on QPS nodes ---
show("2a: Dualmon + 2L parallel on QPS nodes",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0)),
      (2, 0, Inductor(2.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  JJ (1,0), C (1,0), L1 (1,2), QPS (2,0), L2 (2,0)

L2 on QPS nodes (2,0) -> parallel with QPS.
Parallel inductors: each contributes E_L * phi^2 independently.""")

# --- 2b: Dualmon + 2 caps in parallel on JJ nodes ---
show("2b: Dualmon + 2C parallel on JJ nodes",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 0, Capacitor(3.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  JJ (1,0), C1 (1,0), C2 (1,0), L (1,2), QPS (2,0)

Two caps on (1,0) -> parallel: C_total = C1 + C2.
Both on JJ nodes, not QPS nodes -> no Cap||QPS conflict.""")

# --- 2c: Dualmon + inductors in series (extra node) ---
show("2c: Dualmon + L series: L(1,3)+L(3,2) instead of L(1,2)",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 3, Inductor(1.0)),
      (3, 2, Inductor(3.0)),
      (2, 0, PhaseSlip(3.0))],
     """Topology (4 nodes: 0=ground, 1, 2, 3):
  JJ (1,0), C (1,0), L1 (1,3), L2 (3,2), QPS (2,0)

Node 3 has only inductors, no cap -> no charge variable for node 3.
Kirchhoff (K) eliminates it. Schur complement integrates out the
non-dynamical flux. Result: L in series, E_L_eff = E_L1*E_L2/(E_L1+E_L2).""")

# --- 2d: Dualmon + caps in series between JJ and ground ---
show("2d: Dualmon + C series: C(1,3)+C(3,0) coupling JJ to ground",
     [(1, 0, Junction(5.0)),
      (1, 3, Capacitor(1.0)),
      (3, 0, Capacitor(3.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0))],
     """Topology (4 nodes: 0=ground, 1, 2, 3):
  JJ (1,0), C1 (1,3), C2 (3,0), L (1,2), QPS (2,0)

Node 3 has only caps, no inductor -> no flux variable for node 3.
Kirchhoff (K) eliminates it. Schur complement integrates out the
non-dynamical charge. Result: caps in series, E_C_eff = E_C1 + E_C2.""")

# --- 2e: Dualmon + extra L+C parallel branch (ERROR: Cap||QPS) ---
show("2e: Dualmon + extra LC on QPS nodes -- ERROR: Cap||QPS",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0)),
      (2, 0, Inductor(2.0)),
      (2, 0, Capacitor(2.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  JJ (1,0), C1 (1,0), L1 (1,2), QPS (2,0), L2 (2,0), C2 (2,0)

C2 on (2,0) = same nodes as QPS -> Cap||QPS conflict.
Even though L2 is also on those nodes, the Cap||QPS check catches
it before Kirchhoff runs.
Expected: ValueError (Capacitor in parallel with PhaseSlip).""")


# =====================================================================
# SECTION 3: 4-node networks with JJ and QPS
# =====================================================================

print()
print(SEP)
print("SECTION 3: 4-NODE NETWORKS WITH JJ AND QPS")
print("=" * 70)

# --- 3a: JJ-QPS-JJ-QPS ring ---
show("3a: Ring: JJ(0,1) + QPS(1,2) + JJ(2,3) + QPS(3,0)",
     [(0, 1, Junction(5.0)),
      (0, 1, Capacitor(1.0)),
      (1, 2, PhaseSlip(3.0)),
      (1, 2, Inductor(1.0)),
      (2, 3, Junction(7.0)),
      (2, 3, Capacitor(2.0)),
      (3, 0, PhaseSlip(2.0)),
      (3, 0, Inductor(3.0))],
     """Topology (4 nodes: 0=ground, 1, 2, 3):
  JJ1+C1 (0,1), QPS1+L1 (1,2), JJ2+C2 (2,3), QPS2+L2 (3,0)

Alternating JJ-QPS ring. Each JJ has companion C, each QPS has companion L.
Cap and QPS are on DIFFERENT node pairs -> no Cap||QPS conflict.""")

# --- 3b: Star with JJ center and QPS arms ---
show("3b: Star: JJ+C at center(0), QPS+L arms to 1,2,3",
     [(0, 1, Junction(5.0)),
      (0, 1, Capacitor(1.0)),
      (0, 2, PhaseSlip(3.0)),
      (0, 2, Inductor(1.0)),
      (0, 3, PhaseSlip(2.0)),
      (0, 3, Inductor(2.0)),
      (1, 2, Capacitor(0.5)),
      (2, 3, Capacitor(0.5))],
     """Topology (4 nodes: 0=ground, 1, 2, 3):
  JJ+C1 (0,1), QPS1+L1 (0,2), QPS2+L2 (0,3), C3 (1,2), C4 (2,3)

Star topology: JJ at center, QPS+L arms, coupling caps between arms.
Coupling caps are on nodes (1,2) and (2,3) -- no QPS on those nodes.""")

# --- 3c: Two coupled dualmons ---
show("3c: Two coupled dualmons: dualmon(0,1,2) + dualmon(0,3,4) + coupling",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0)),
      (3, 0, Junction(7.0)),
      (3, 0, Capacitor(2.0)),
      (3, 4, Inductor(2.0)),
      (4, 0, PhaseSlip(2.0)),
      (2, 4, Capacitor(0.5))],
     """Topology (5 nodes: 0=ground, 1, 2, 3, 4):
  Dualmon 1: JJ1+C1 (1,0), L1 (1,2), QPS1 (2,0)
  Dualmon 2: JJ2+C2 (3,0), L2 (3,4), QPS2 (4,0)
  Coupling:  Cg (2,4) between QPS nodes

Two full dualmons sharing ground, coupled by Cg(2,4).
Cg is on nodes (2,4) -- no QPS on those exact nodes as a pair.""")

# --- 3d: 4-node fully connected with mixed elements ---
show("3d: 4-node mesh: JJ(0,1), JJ(2,3), QPS(1,2), QPS(0,3), L+C coupling",
     [(0, 1, Junction(5.0)),
      (0, 1, Capacitor(1.0)),
      (2, 3, Junction(7.0)),
      (2, 3, Capacitor(2.0)),
      (1, 2, PhaseSlip(3.0)),
      (1, 2, Inductor(1.0)),
      (0, 3, PhaseSlip(2.0)),
      (0, 3, Inductor(2.0)),
      (0, 2, Capacitor(0.5)),
      (1, 3, Inductor(3.0))],
     """Topology (4 nodes: 0=ground, 1, 2, 3):
  JJ1+C1 (0,1), JJ2+C2 (2,3), QPS1+L1 (1,2), QPS2+L3 (0,3)
  Coupling: C3 (0,2), L4 (1,3)

Fully connected 4-node network with mixed JJ and QPS elements.
Each nonlinear element has its companion on the same nodes.
Coupling C and L are on nodes without QPS -> no Cap||QPS conflict.""")


print(SEP)
print("DONE")
