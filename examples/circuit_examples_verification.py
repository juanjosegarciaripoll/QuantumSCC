"""

"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip

SEP = "\n" + "=" * 70


# ─────────────────────────────────────────────────────────────────────
# 1) Basic LC oscillator  (1 mode)
# ─────────────────────────────────────────────────────────────────────
print(SEP)
print("CIRCUIT 1: Basic LC oscillator")

c1 = Circuit([
    (0, 1, Capacitor(1.0)),
    (0, 1, Inductor(1.0)),
])
c1.symbolic_hamiltonian_expression(verbose=False)
c1.Hamiltonian_expression(verbose=False)


# ─────────────────────────────────────────────────────────────────────
# 2) N caps parallel + N inductors parallel  (1 mode)
# ─────────────────────────────────────────────────────────────────────
print(SEP)
print("CIRCUIT 2: Parallel caps + parallel inductors (1 mode)")

c2 = Circuit([
    (0, 1, Capacitor(1.0)),
    (0, 1, Capacitor(3.0)),
    (0, 1, Inductor(1.0)),
    (0, 1, Inductor(3.0)),
])
c2.symbolic_hamiltonian_expression(verbose=False)
c2.Hamiltonian_expression(verbose=False)


# ─────────────────────────────────────────────────────────────────────
# 3a) Series caps + inductor
# ─────────────────────────────────────────────────────────────────────
print(SEP)
print("CIRCUIT 3a: Series caps (C1, C2) + inductor (L)")

c3a = Circuit([
    (0, 1, Capacitor(1.0)),
    (1, 2, Capacitor(3.0)),
    (0, 2, Inductor(1.0)),
])
c3a.symbolic_hamiltonian_expression(verbose=False)
c3a.Hamiltonian_expression(verbose=False)


# ─────────────────────────────────────────────────────────────────────
# 3b) Cap + series inductors
# ─────────────────────────────────────────────────────────────────────
print(SEP)
print("CIRCUIT 3b: Cap (C) + series inductors (L1, L2)")

c3b = Circuit([
    (0, 2, Capacitor(1.0)),
    (0, 1, Inductor(1.0)),
    (1, 2, Inductor(3.0)),
])
c3b.symbolic_hamiltonian_expression(verbose=False)
c3b.Hamiltonian_expression(verbose=False)


# ─────────────────────────────────────────────────────────────────────
# 4*) Two-mode circuit (3 nodes)
# ─────────────────────────────────────────────────────────────────────
print(SEP)
print("CIRCUIT 4*: Two-mode LC")

c4s = Circuit([
    (1, 0, Capacitor(1.0)),
    (1, 0, Inductor(1.0)),
    (2, 0, Capacitor(1.0)),
    (1, 2, Inductor(1.0)),
])
c4s.symbolic_hamiltonian_expression(verbose=False)
c4s.Hamiltonian_expression(verbose=False)


# ─────────────────────────────────────────────────────────────────────
# 5) Star LC, 3 modes (4 nodes)
# ─────────────────────────────────────────────────────────────────────
print(SEP)
print("CIRCUIT 5: Star LC, 3 modes")

c5 = Circuit([
    (1, 0, Inductor(1.0)),
    (1, 0, Capacitor(1.0)),
    (2, 0, Capacitor(1.0)),
    (3, 0, Capacitor(1.0)),
    (1, 2, Inductor(1.0)),
    (1, 3, Inductor(1.0)),
])
c5.symbolic_hamiltonian_expression(verbose=False)
c5.Hamiltonian_expression(verbose=False)


# ─────────────────────────────────────────────────────────────────────
# A) Nonlinear: 2 JJs + 2 inductors in series
# ─────────────────────────────────────────────────────────────────────
print(SEP)
print("CIRCUIT A: Two JJs + series inductors")

cA = Circuit([
    (1, 0, Junction(1.0)),
    (1, 0, Capacitor(1.0)),
    (2, 0, Junction(1.0)),
    (2, 0, Capacitor(1.0)),
    (1, 3, Inductor(1.0)),
    (3, 2, Inductor(1.0)),
])
cA.symbolic_hamiltonian_expression(verbose=False)
cA.Hamiltonian_expression(verbose=False)


# ─────────────────────────────────────────────────────────────────────
# B) Complex nonlinear: 4 nodes (0,1,2,3), 2 JJ + 4 L + 3 C
#    Original: all extended (no islands, inductors in every JJ loop)
# ─────────────────────────────────────────────────────────────────────
print(SEP)
print("CIRCUIT B (original): 4-node nonlinear (2 JJ + 4 L + 3 C) — all extended")
print("Topology: JJ1+C1(0,1), JJ2+C2(3,2), L1(0,3), L2(3,1),")
print("          L3+C3(0,2), L4(1,2)")

cB = Circuit([
    (0, 1, Junction(1.0)),
    (0, 1, Capacitor(1.0)),
    (3, 2, Junction(1.0)),
    (3, 2, Capacitor(1.0)),
    (0, 3, Inductor(1.0)),
    (3, 1, Inductor(1.0)),
    (0, 2, Inductor(1.0)),
    (0, 2, Capacitor(1.0)),
    (1, 2, Inductor(1.0)),
])
cB.symbolic_hamiltonian_expression(verbose=False)
cB.Hamiltonian_expression(verbose=False)


# ─────────────────────────────────────────────────────────────────────
# B1) Variant: Replace L1(0,3) and L2(3,1) by capacitors
#     Now a JJ loop has no inductor → 1 compact flux mode (island)
# ─────────────────────────────────────────────────────────────────────
print(SEP)
print("CIRCUIT B1: Replace L1,L2 by caps — 1 compact mode (island)")
print("Topology: JJ1+C1(0,1), JJ2+C2(3,2), C4(0,3), C5(3,1),")
print("          L3+C3(0,2), L4(1,2)")

cB1 = Circuit([
    (0, 1, Junction(1.0)),
    (0, 1, Capacitor(1.0)),
    (3, 2, Junction(1.0)),
    (3, 2, Capacitor(1.0)),
    (0, 3, Capacitor(1.0)),
    (3, 1, Capacitor(1.0)),
    (0, 2, Inductor(1.0)),
    (0, 2, Capacitor(1.0)),
    (1, 2, Inductor(1.0)),
])
cB1.symbolic_hamiltonian_expression(verbose=False)
cB1.Hamiltonian_expression(verbose=False)


# ─────────────────────────────────────────────────────────────────────
# B2) Variant: Replace L2(3,1) and L4(1,2) by capacitors
#     Different JJ becomes compact
# ─────────────────────────────────────────────────────────────────────
print(SEP)
print("CIRCUIT B2: Replace L2,L4 by caps — 1 compact mode (island)")
print("Topology: JJ1+C1(0,1), JJ2+C2(3,2), L1(0,3), C4(3,1),")
print("          L3+C3(0,2), C5(1,2)")

cB2 = Circuit([
    (0, 1, Junction(1.0)),
    (0, 1, Capacitor(1.0)),
    (3, 2, Junction(1.0)),
    (3, 2, Capacitor(1.0)),
    (0, 3, Inductor(1.0)),
    (3, 1, Capacitor(1.0)),
    (0, 2, Inductor(1.0)),
    (0, 2, Capacitor(1.0)),
    (1, 2, Capacitor(1.0)),
])
cB2.symbolic_hamiltonian_expression(verbose=False)
cB2.Hamiltonian_expression(verbose=False)

print(SEP)
print("DONE")
