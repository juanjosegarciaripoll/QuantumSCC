# Your First Circuit with QuantumSCC

QuantumSCC quantizes superconducting circuits automatically. You describe
your circuit as a graph — nodes and branches — and the library derives the
quantum Hamiltonian via Faddeev-Jackiw / Darboux reduction.

---

## Installation

```bash
pip install -e .
```

Then, in any Python script or notebook:

```python
from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip
```

These five names are everything you need to define any circuit.

---

## The circuit model

A circuit is a **graph**:

- **Nodes** — integers (any value, any numbering)
- **Branches** — circuit elements connecting two nodes

You define it as a list of `(node_a, node_b, element)` tuples:

```python
circuit = Circuit([
    (0, 1, element_A),
    (0, 1, element_B),   # parallel to A
    (1, 2, element_C),   # in series
])
```

That is all. No geometry, no coordinates, no matrix construction by hand.

---

## Elements and units

Every element accepts a numeric `value` and a `unit` string.
Units can be **physical** (Farads, Henries) or **energy** (GHz, MHz):

| Element | Physical units | Energy units |
|---|---|---|
| `Capacitor` | `'fF'`, `'pF'`, `'nF'` | `'GHz'`, `'MHz'`, `'THz'` (as E_C) |
| `Inductor` | `'fH'`, `'pH'`, `'nH'`, `'uH'` | `'GHz'`, `'MHz'` (as E_L) |
| `Junction` | — | `'GHz'`, `'MHz'`, `'THz'` (as E_J) |
| `PhaseSlip` | — | `'GHz'`, `'MHz'`, `'THz'` (as E_P) |

Each element is a **single branch**. Nonlinear elements (`Junction`, `PhaseSlip`)
carry only their cosine energy. If you need a companion (capacitor for JJ,
inductor for QPS), add it as a **separate branch** on the same nodes.

---

## Example 1 — LC oscillator (linear)

The simplest complete circuit: one inductor and one capacitor in parallel.

```python
C = Capacitor(value=1, unit='pF')   # 1 pF
L = Inductor(value=1,  unit='nH')   # 1 nH

circuit = Circuit([(0, 1, L), (0, 1, C)])

# Harmonic frequency in GHz
import numpy as np
H = circuit.extended_quantum_hamiltonian
omega = H[0, 0].real
print(f"omega / 2pi = {omega:.4f} GHz")  # -> 31.6228 GHz  (= 1/sqrt(LC) x 1e-9)
```

**What you can read:**

```python
circuit.quadratic_hamiltonian          # 2x2 numpy array: H_quadratic
circuit.extended_quantum_hamiltonian   # diagonal matrix of mode frequencies
circuit.no_independent_variables       # 2 (one conjugate pair: phi, q)
circuit.Hamiltonian_expression()       # prints a symbolic expression
```

---

## Example 2 — Transmon (single Josephson junction)

A transmon is a JJ shunted by a capacitor. Each is a separate branch
on the same nodes:

```python
C = Capacitor(value=70, unit='fF')
J = Junction(value=13.4, unit='GHz')

circuit = Circuit([
    (0, 1, J),   # Josephson junction
    (0, 1, C),   # parallel capacitor (shunt)
])

circuit.Hamiltonian_expression()
# H/hbar (GHz) = + 27.819 (n_c1)^2 - 13.400 cos(v_1 xi_phi)
```

The Hamiltonian has:
- A quadratic charging term `4 E_C n^2` (here `4 E_C ~ 27.8 GHz`)
- A nonlinear Josephson term `-E_J cos(phi_c)` where `phi_c` is the **compact flux**

```python
circuit.no_final_compact_flux    # 1 — one compact flux variable
circuit.vector_JJ                # coupling vector for the cos(phi) term
```

---

## Example 3 — Fluxonium (junction + inductor)

Adding an inductor in parallel with the junction makes the junction flux
**extended** — it can now wind over the full real line, not just a circle.
This is the fluxonium qubit.

```python
C = Capacitor(value=5,   unit='fF')
J = Junction(value=8.0,  unit='GHz')
L = Inductor(value=400,  unit='nH')

circuit = Circuit([
    (0, 1, J),   # Josephson junction
    (0, 1, C),   # parallel capacitor
    (0, 1, L),   # parallel inductor (superinductor)
])

circuit.Hamiltonian_expression()
# H/hbar (GHz) = + 31.6 [(phi_e1)^2 + (n_e1)^2] - 8.0 cos(v_1 xi_phi)
```

The `e` subscript means *extended*: `phi_e1` is an oscillator mode, not a
compact variable. The inductor kills the compact flux by adding a loop
(see Eq. 42 in PRX 2025).

---

## Example 4 — Dual-transmon (Phase Slip element)

`PhaseSlip` is the electromagnetic dual of `Junction`:

| Junction | PhaseSlip |
|---|---|
| Compact flux phi in S^1 | Compact charge q in S^1 |
| Energy: -E_J cos(phi) | Energy: -E_P cos(q) |
| Needs parallel capacitor | Needs parallel inductor |

```python
L = Inductor(value=1,   unit='nH')
P = PhaseSlip(value=5.0, unit='GHz')

circuit = Circuit([
    (0, 1, P),   # phase-slip element
    (0, 1, L),   # parallel inductor (companion)
])

circuit.Hamiltonian_expression()
# H/hbar (GHz) = + E_L (psi_c1)^2 - 5.000 cos(u_1 xi_q)

circuit.no_final_compact_charge   # 1 — one compact charge variable
circuit.vector_QPS                # coupling vector for the cos(q) term
```

---

## Example 5 — Mixed circuit: JJ + QPS chain

JJ and QPS elements can coexist in the same circuit, on **different** nodes.

```python
C = Capacitor(value=1, unit='GHz')
J = Junction(value=1,  unit='GHz')
L = Inductor(value=1,  unit='GHz')
P = PhaseSlip(value=1, unit='GHz')

# JJ+C on branch (0,1), QPS+L on branch (1,2)
circuit = Circuit([
    (0, 1, J),
    (0, 1, C),
    (1, 2, P),
    (1, 2, L),
])

circuit.Hamiltonian_expression()

circuit.no_final_compact_flux    # 1
circuit.no_final_compact_charge  # 1
```

Both the compact flux (from J) and the compact charge (from P) appear
as independent nonlinear degrees of freedom.

**Important:** Capacitor and PhaseSlip must NOT be on the same nodes.
This creates a nonlinear KVL constraint (Kepler equation) that cannot
be solved within the Faddeev-Jackiw framework.

---

## Example 6 — Multiple modes: 2 JJ in series

Two junctions in series (with capacitors and a closing inductor) give
multiple modes:

```python
circuit = Circuit([
    (0, 1, Junction(1, 'GHz')),
    (0, 1, Capacitor(1, 'GHz')),
    (1, 2, Junction(1, 'GHz')),
    (1, 2, Capacitor(1, 'GHz')),
    (0, 2, Inductor(1, 'GHz')),
])

circuit.quadratic_hamiltonian.shape   # (4, 4) — 2 modes x 2 variables each
circuit.vector_JJ.shape               # (4, 2) — one column per junction
```

---

## Reading the results

| Attribute | Type | Description |
|---|---|---|
| `circuit.quadratic_hamiltonian` | `ndarray (n x n)` | Quadratic part of H (pre-diagonalization) |
| `circuit.extended_quantum_hamiltonian` | `ndarray (2k x 2k)` | Diagonal: harmonic mode frequencies |
| `circuit.vector_JJ` | `ndarray (n x nJJ)` | Coupling vectors for each junction's cos(phi) |
| `circuit.vector_QPS` | `ndarray (n x nQPS)` | Coupling vectors for each phase-slip's cos(q) |
| `circuit.no_final_compact_flux` | `int` | Number of compact flux modes |
| `circuit.no_final_compact_charge` | `int` | Number of compact charge modes |
| `circuit.Hamiltonian_expression()` | `(prints)` | Numerical Hamiltonian in GHz units |
| `circuit.symbolic_hamiltonian_expression()` | `(prints)` | Symbolic Hamiltonian with E_C, E_L, E_J |
| `circuit.FS_quadratic_hamiltonian_phiq` | `ndarray` | H in the flux-charge (phi,q) basis |
| `circuit.FS_quadratic_hamiltonian_an` | `ndarray` | H in the creation-annihilation (a+,a) basis |

---

## Common mistakes

**1. Forgetting the companion element:**
```python
# A bare Junction works (e.g., bare dualmon), but won't have
# quadratic energy. For a transmon, add a Capacitor:
J = Junction(value=10, unit='GHz')
C = Capacitor(value=1, unit='pF')
circuit = Circuit([(0, 1, J), (0, 1, C)])   # transmon
```

**2. Capacitor on the same nodes as PhaseSlip:**
```python
# This raises ValueError — Kepler equation, not solvable:
Circuit([(0, 1, PhaseSlip(5, 'GHz')), (0, 1, Capacitor(1, 'pF'))])
# Solution: remove the capacitor, or add an inductor on those nodes
```

**3. Wrong unit for Junction or PhaseSlip** (only frequency units are valid):
```python
Junction(value=1, unit='pF')    # ValueError: unit not correct
```

**4. A disconnected circuit:**
```python
Circuit([(0, 1, Capacitor(1,'pF')), (2, 3, Inductor(1,'nH'))])
# ValueError: disconnected circuit
```

**5. Reusing the same element object on multiple branches:**
```python
J = Junction(1, 'GHz')
Circuit([(0, 1, J), (0, 1, J)])   # same object twice — create two instances
```
Always instantiate a fresh element for each branch.

---

## Full working example (copy-paste ready)

```python
from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip
import numpy as np

# Fluxonium: JJ shunted by a superinductor
C = Capacitor(value=5,   unit='fF')
J = Junction(value=8.0,  unit='GHz')
L = Inductor(value=400,  unit='nH')

circuit = Circuit([
    (0, 1, J),   # Josephson junction
    (0, 1, C),   # parallel capacitor
    (0, 1, L),   # superinductor
])

print(f"Modes          : {circuit.no_independent_variables // 2}")
print(f"Compact flux   : {circuit.no_final_compact_flux}")
print(f"Compact charge : {circuit.no_final_compact_charge}")
print()
circuit.Hamiltonian_expression()
```

Output:
```
Modes          : 1
Compact flux   : 0
Compact charge : 0

Numerical Hamiltonian:
H/hbar (GHz) = + 31.6 [(phi_e1)^2 + (n_e1)^2]  - 8.000 cos(v_1 xi_phi)

JJ coupling vectors v (flux space):
v_1 = [-0.175  0.   ]
```
