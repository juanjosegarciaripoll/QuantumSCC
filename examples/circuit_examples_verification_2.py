"""
Verification examples — Part 2: Dualmon variants and 4-node JJ+QPS networks.

Explores:
  - Dualmon with dummy capacitors (eliminated by Kirchhoff/Schur)
  - Dualmon with series/parallel L and C combinations
  - 4-node network with JJ and QPS
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip
import numpy as np

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
    print(f"Modes: nCF={nCF}, nCC={nCC}, nEF={nEF} → {nF} pairs")

    V = c.geom.V
    V_phi = V[:nF, :nF] if nF > 0 else np.array([[]])
    print(f"V_Phi = I? {np.allclose(V_phi, np.eye(nF)) if nF > 0 else 'N/A'}")
    print()

    c.symbolic_hamiltonian_expression(verbose=False)
    c.Hamiltonian_expression(verbose=False)
    return c


# =====================================================================
# SECTION 1: Dualmon with dummy capacitors
# =====================================================================

print(SEP)
print("SECTION 1: DUALMON WITH DUMMY CAPACITORS")
print("=" * 70)
print("""
The basic dualmon is JJ(0,1) + QPS(0,1): two nonlinear elements,
no quadratic energy, 0 modes. Adding a capacitor or inductor gives
it dynamics.

A "dummy" capacitor on the SAME nodes as the QPS (0,1) makes the
QPS charge extended (kills the compact charge). This is because:
  - QPS creates a compact charge (periodic S¹)
  - Capacitor on same nodes adds Q²/(2C) → extends charge to R
  - The compact charge is "absorbed" by the capacitor
  - kcut_suppressed = True in the code

Similarly, a capacitor on DIFFERENT nodes is truly independent
and does NOT kill the compact charge.
""")

# --- 1a: Bare dualmon (no dynamics) ---
show("1a: Bare dualmon — JJ + QPS (no quadratic energy)",
     [(0, 1, Junction(5.0)),
      (0, 1, PhaseSlip(3.0))],
     """Topology (2 nodes: 0=ground, 1):
  Node 0 (ground): —
  Node 1: JJ(0,1), QPS(0,1)

Full element list:
  JJ  (0,1) — Josephson junction, E_J=5 GHz
  QPS (0,1) — phase-slip element, E_P=3 GHz

No C, no L → no quadratic H. Only cos(φ) + cos(q).""")

# --- 1b: Dualmon + C on same nodes as QPS (dummy — kills compact charge) ---
show("1b: Dualmon + C on QPS nodes (0,1) — DUMMY cap kills compact charge",
     [(0, 1, Junction(5.0)),
      (0, 1, PhaseSlip(3.0)),
      (0, 1, Capacitor(1.0))],
     """Topology (2 nodes: 0=ground, 1):
  Node 0 (ground): —
  Node 1: JJ(0,1), QPS(0,1), C(0,1)

Full element list:
  JJ  (0,1) — Josephson junction, E_J=5 GHz
  QPS (0,1) — phase-slip element, E_P=3 GHz
  C   (0,1) — capacitor, E_C=1 GHz  ← DUMMY: same nodes as QPS

C on (0,1) = same nodes as QPS → the cap extends the QPS charge from S¹ to R.
In the topology step, kcut_suppressed=True: the compact charge column of K
is suppressed. The cap energy enters the Schur complement of the Hamiltonian.
Result: nCC=0 (compact charge killed).""")

# --- 1c: Dualmon + L on same nodes (gives oscillator) ---
show("1c: Dualmon + L on (0,1) — inductor gives dynamics",
     [(0, 1, Junction(5.0)),
      (0, 1, PhaseSlip(3.0)),
      (0, 1, Inductor(1.0))],
     """Topology (2 nodes: 0=ground, 1):
  Node 0 (ground): —
  Node 1: JJ(0,1), QPS(0,1), L(0,1)

Full element list:
  JJ  (0,1) — Josephson junction, E_J=5 GHz
  QPS (0,1) — phase-slip element, E_P=3 GHz
  L   (0,1) — inductor, E_L=1 GHz

L on same nodes adds Φ²/(2L). QPS keeps compact charge (no cap kills it).
L in JJ loop → kills JJ compact flux.""")

# --- 1d: Full dualmon (JJ+C on node pair, L series, QPS on other pair) ---
show("1d: Full dualmon — JJ+C(1,0), L(1,2), QPS(2,0)",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  Node 0 (ground): —
  Node 1: JJ(1,0), C(1,0), L(1,2)
  Node 2: L(1,2), QPS(2,0)

Full element list:
  JJ  (1,0) — Josephson junction, E_J=5 GHz
  C   (1,0) — capacitor on JJ nodes, E_C=1 GHz
  L   (1,2) — inductor connecting JJ side to QPS side
  QPS (2,0) — phase-slip element, E_P=3 GHz

Classic dualmon from Le et al.
L in JJ loop (0→1→2→0) → kills JJ compact flux (nCF=0).
No cap on QPS nodes (2,0) → QPS compact charge depends on topology.""")

# --- 1e: Full dualmon + dummy C on QPS nodes ---
show("1e: Full dualmon + dummy C on QPS nodes (2,0)",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0)),
      (2, 0, Capacitor(2.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  Node 0 (ground): —
  Node 1: JJ(1,0), C1(1,0), L(1,2)
  Node 2: L(1,2), QPS(2,0), C2(2,0)  ← C2 is DUMMY (same nodes as QPS)

Full element list:
  JJ  (1,0) — Josephson junction, E_J=5 GHz
  C1  (1,0) — capacitor on JJ nodes, E_C=1 GHz
  L   (1,2) — inductor connecting JJ side to QPS side
  QPS (2,0) — phase-slip element, E_P=3 GHz
  C2  (2,0) — DUMMY capacitor on QPS nodes, E_C=2 GHz

C2 on (2,0) = same as QPS → kills compact charge.
QPS becomes fully decoupled → ERROR expected.""")

# --- 1f: Full dualmon + C on DIFFERENT nodes (not dummy) ---
show("1f: Full dualmon + C on different nodes (1,0) — NOT dummy",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0)),
      (1, 0, Capacitor(2.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  Node 0 (ground): —
  Node 1: JJ(1,0), C1(1,0), C2(1,0), L(1,2)
  Node 2: L(1,2), QPS(2,0)

Full element list:
  JJ  (1,0) — Josephson junction, E_J=5 GHz
  C1  (1,0) — capacitor on JJ nodes, E_C=1 GHz
  C2  (1,0) — extra capacitor on JJ nodes, E_C=2 GHz  ← NOT dummy
  L   (1,2) — inductor connecting JJ side to QPS side
  QPS (2,0) — phase-slip element, E_P=3 GHz

C2 on (1,0) = JJ nodes, NOT QPS nodes → does NOT kill QPS compact charge.
Parallel caps on JJ nodes: C_T = C1+C2.""")


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
  Node 0 (ground): —
  Node 1: JJ(1,0), C(1,0), L1(1,2)
  Node 2: L1(1,2), QPS(2,0), L2(2,0)

Full element list:
  JJ  (1,0) — Josephson junction, E_J=5 GHz
  C   (1,0) — capacitor on JJ nodes, E_C=1 GHz
  L1  (1,2) — inductor, E_L=1 GHz
  QPS (2,0) — phase-slip element, E_P=3 GHz
  L2  (2,0) — inductor parallel with QPS, E_L=2 GHz

L2 on QPS nodes (2,0) → parallel with QPS.
Parallel inductors: each contributes Φ²/(2L_i) independently.""")

# --- 2b: Dualmon + 2 caps in parallel on JJ nodes ---
show("2b: Dualmon + 2C parallel on JJ nodes",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 0, Capacitor(3.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  Node 0 (ground): —
  Node 1: JJ(1,0), C1(1,0), C2(1,0), L(1,2)
  Node 2: L(1,2), QPS(2,0)

Full element list:
  JJ  (1,0) — Josephson junction, E_J=5 GHz
  C1  (1,0) — capacitor, E_C=1 GHz
  C2  (1,0) — capacitor, E_C=3 GHz
  L   (1,2) — inductor, E_L=1 GHz
  QPS (2,0) — phase-slip element, E_P=3 GHz

Two caps on (1,0) → parallel: C_T = C1+C2.
E_C_eff = harmonic mean of E_C1, E_C2.""")

# --- 2c: Dualmon + inductors in series (extra node) ---
show("2c: Dualmon + L series: L(1,3)+L(3,2) instead of L(1,2)",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 3, Inductor(1.0)),
      (3, 2, Inductor(3.0)),
      (2, 0, PhaseSlip(3.0))],
     """Topology (4 nodes: 0=ground, 1, 2, 3):
  Node 0 (ground): —
  Node 1: JJ(1,0), C(1,0), L1(1,3)
  Node 2: L2(3,2), QPS(2,0)
  Node 3: L1(1,3), L2(3,2)  ← only inductors, no cap → intermediate, eliminated

Full element list:
  JJ  (1,0) — Josephson junction, E_J=5 GHz
  C   (1,0) — capacitor on JJ nodes, E_C=1 GHz
  L1  (1,3) — inductor (first half of series), E_L=1 GHz
  L2  (3,2) — inductor (second half of series), E_L=3 GHz
  QPS (2,0) — phase-slip element, E_P=3 GHz

Node 3 has only inductors, no cap → no charge variable for node 3.
Kirchhoff (K) eliminates it: the kernel of F does not include node 3.
Then Schur complement integrates out the non-dynamical flux.
Result: L in series, E_L_eff = E_L1·E_L2/(E_L1+E_L2).""")

# --- 2d: Dualmon + caps in series between JJ and ground ---
show("2d: Dualmon + C series: C(1,3)+C(3,0) coupling JJ to ground",
     [(1, 0, Junction(5.0)),
      (1, 3, Capacitor(1.0)),
      (3, 0, Capacitor(3.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0))],
     """Topology (4 nodes: 0=ground, 1, 2, 3):
  Node 0 (ground): C2(3,0)
  Node 1: JJ(1,0), C1(1,3), L(1,2)
  Node 2: L(1,2), QPS(2,0)
  Node 3: C1(1,3), C2(3,0)  ← only caps, no inductor → intermediate, eliminated

Full element list:
  JJ  (1,0) — Josephson junction, E_J=5 GHz
  C1  (1,3) — capacitor in series (first half), E_C=1 GHz
  C2  (3,0) — capacitor in series (second half), E_C=3 GHz
  L   (1,2) — inductor connecting JJ side to QPS side, E_L=1 GHz
  QPS (2,0) — phase-slip element, E_P=3 GHz

Node 3 has only caps, no inductor → no flux variable for node 3.
Kirchhoff (K) eliminates it: the kernel of F does not include node 3.
Then Schur complement integrates out the non-dynamical charge.
Result: caps in series, E_C_eff = E_C1 + E_C2.""")

# --- 2e: Dualmon + extra L+C parallel branch ---
show("2e: Dualmon + extra LC branch in parallel",
     [(1, 0, Junction(5.0)),
      (1, 0, Capacitor(1.0)),
      (1, 2, Inductor(1.0)),
      (2, 0, PhaseSlip(3.0)),
      (2, 0, Inductor(2.0)),
      (2, 0, Capacitor(2.0))],
     """Topology (3 nodes: 0=ground, 1, 2):
  Node 0 (ground): —
  Node 1: JJ(1,0), C1(1,0), L1(1,2)
  Node 2: L1(1,2), QPS(2,0), L2(2,0), C2(2,0)

Full element list:
  JJ  (1,0) — Josephson junction, E_J=5 GHz
  C1  (1,0) — capacitor on JJ nodes, E_C=1 GHz
  L1  (1,2) — inductor, E_L=1 GHz
  QPS (2,0) — phase-slip element, E_P=3 GHz
  L2  (2,0) — extra inductor parallel with QPS, E_L=2 GHz
  C2  (2,0) — DUMMY capacitor on QPS nodes, E_C=2 GHz

C2 on (2,0) = same nodes as QPS → kills compact charge (dummy).
L2 adds inductance to QPS side.""")


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
  Node 0 (ground): QPS2(3,0), L2(3,0)
  Node 1: JJ1(0,1), C1(0,1), QPS1(1,2), L1(1,2)
  Node 2: QPS1(1,2), L1(1,2), JJ2(2,3), C2(2,3)
  Node 3: JJ2(2,3), C2(2,3), QPS2(3,0), L2(3,0)

Full element list:
  JJ1  (0,1) — Josephson junction, E_J=5 GHz
  C1   (0,1) — capacitor on JJ1 nodes, E_C=1 GHz
  QPS1 (1,2) — phase-slip element, E_P=3 GHz
  L1   (1,2) — inductor on QPS1 nodes, E_L=1 GHz
  JJ2  (2,3) — Josephson junction, E_J=7 GHz
  C2   (2,3) — capacitor on JJ2 nodes, E_C=2 GHz
  QPS2 (3,0) — phase-slip element, E_P=2 GHz
  L2   (3,0) — inductor on QPS2 nodes, E_L=3 GHz

Alternating JJ-QPS ring. Each JJ has C, each QPS has L.
Inductors in JJ loops may kill compact flux.""")

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
  Node 0 (ground): —
  Node 1: JJ(0,1), C1(0,1), C3(1,2)
  Node 2: QPS1(0,2), L1(0,2), C3(1,2), C4(2,3)
  Node 3: QPS2(0,3), L2(0,3), C4(2,3)

Full element list:
  JJ   (0,1) — Josephson junction, E_J=5 GHz
  C1   (0,1) — capacitor on JJ nodes, E_C=1 GHz
  QPS1 (0,2) — phase-slip element, E_P=3 GHz
  L1   (0,2) — inductor on QPS1 nodes, E_L=1 GHz
  QPS2 (0,3) — phase-slip element, E_P=2 GHz
  L2   (0,3) — inductor on QPS2 nodes, E_L=2 GHz
  C3   (1,2) — coupling capacitor between arms, E_C=0.5 GHz
  C4   (2,3) — coupling capacitor between arms, E_C=0.5 GHz

Star topology: JJ at center, QPS+L arms, coupling caps between arms.""")

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
  Node 0 (ground): —
  Node 1: JJ1(1,0), C1(1,0), L1(1,2)
  Node 2: L1(1,2), QPS1(2,0), Cg(2,4)
  Node 3: JJ2(3,0), C2(3,0), L2(3,4)
  Node 4: L2(3,4), QPS2(4,0), Cg(2,4)

Full element list:
  JJ1  (1,0) — Josephson junction, E_J=5 GHz
  C1   (1,0) — capacitor on JJ1 nodes, E_C=1 GHz
  L1   (1,2) — inductor, E_L=1 GHz
  QPS1 (2,0) — phase-slip element, E_P=3 GHz
  JJ2  (3,0) — Josephson junction, E_J=7 GHz
  C2   (3,0) — capacitor on JJ2 nodes, E_C=2 GHz
  L2   (3,4) — inductor, E_L=2 GHz
  QPS2 (4,0) — phase-slip element, E_P=2 GHz
  Cg   (2,4) — coupling capacitor between QPS nodes, E_C=0.5 GHz

Two full dualmons sharing ground, coupled by Cg(2,4).""")

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
  Node 0 (ground): QPS2(0,3), L3(0,3), C3(0,2)
  Node 1: JJ1(0,1), C1(0,1), QPS1(1,2), L1(1,2), L4(1,3)
  Node 2: QPS1(1,2), L1(1,2), JJ2(2,3), C2(2,3), C3(0,2)
  Node 3: JJ2(2,3), C2(2,3), QPS2(0,3), L3(0,3), L4(1,3)

Full element list:
  JJ1  (0,1) — Josephson junction, E_J=5 GHz
  C1   (0,1) — capacitor on JJ1 nodes, E_C=1 GHz
  JJ2  (2,3) — Josephson junction, E_J=7 GHz
  C2   (2,3) — capacitor on JJ2 nodes, E_C=2 GHz
  QPS1 (1,2) — phase-slip element, E_P=3 GHz
  L1   (1,2) — inductor on QPS1 nodes, E_L=1 GHz
  QPS2 (0,3) — phase-slip element, E_P=2 GHz
  L3   (0,3) — inductor on QPS2 nodes, E_L=2 GHz
  C3   (0,2) — coupling capacitor, E_C=0.5 GHz
  L4   (1,3) — coupling inductor, E_L=3 GHz

Fully connected 4-node network with mixed JJ and QPS elements.""")


print(SEP)
print("DONE")
