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

`Junction` always requires a parallel capacitor (`cap=`).
`PhaseSlip` always requires a parallel inductor (`ind=`).

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
print(f"ω / 2π = {omega:.4f} GHz")  # → 31.6228 GHz  (= 1/√(LC) × 1e-9)
```

**What you can read:**

```python
circuit.quadratic_hamiltonian          # 2×2 numpy array: H_quadratic
circuit.extended_quantum_hamiltonian   # diagonal matrix of mode frequencies
circuit.no_independent_variables       # 2 (one conjugate pair: φ, q)
circuit.Hamiltonian_expression()       # prints a symbolic expression
```

---

## Example 2 — Transmon (single Josephson junction)

```python
C = Capacitor(value=70, unit='fF')
J = Junction(value=13.4, unit='GHz', cap=C)

circuit = Circuit([(0, 1, J)])

circuit.Hamiltonian_expression()
# H/ℏ (GHz) = + 27.819 (n_c1)² − 13.400 cos(v_1 ξφ)
```

The Hamiltonian has:
- A quadratic charging term `4 E_C n²` (here `4 E_C ≈ 27.8 GHz`)
- A nonlinear Josephson term `−E_J cos(φ_c)` where `φ_c` is the **compact flux**

```python
circuit.no_final_compact_flux    # 1 — one compact flux variable
circuit.vector_JJ                # coupling vector for the cos(φ) term
```

---

## Example 3 — Fluxonium (junction + inductor)

Adding an inductor in parallel with the junction makes the junction flux
**extended** — it can now wind over the full real line, not just a circle.
This is the fluxonium qubit.

```python
C = Capacitor(value=5,   unit='fF')
J = Junction(value=8.0,  unit='GHz', cap=C)
L = Inductor(value=400,  unit='nH')

circuit = Circuit([(0, 1, J), (0, 1, L)])

circuit.Hamiltonian_expression()
# H/ℏ (GHz) = + 31.6 [(ϕ_e1)² + (n_e1)²] − 8.0 cos(v_1 ξφ)
```

The `e` subscript means *extended*: `ϕ_e1` is an oscillator mode, not a
compact variable. The quadratic part is now the harmonic oscillator term.

---

## Example 4 — Dual-transmon (Phase Slip element)

`PhaseSlip` is the electromagnetic dual of `Junction`:

| Junction | PhaseSlip |
|---|---|
| Compact flux φ ∈ S¹ | Compact charge q ∈ S¹ |
| Energy: −E_J cos(φ) | Energy: −E_P cos(q) |
| Needs parallel capacitor | Needs parallel inductor |

```python
L = Inductor(value=1,   unit='nH')
P = PhaseSlip(value=5.0, unit='GHz', ind=L)

circuit = Circuit([(0, 1, P)])

circuit.Hamiltonian_expression()
# H/ℏ (GHz) = − 5.000 cos(u_1 ξq)
#
# QPS coupling vectors u (charge space):
# u_1 = [ 0. -1.]

circuit.no_final_compact_charge   # 1 — one compact charge variable
circuit.vector_QPS                # coupling vector for the cos(q) term
```

---

## Example 5 — Mixed circuit: JJ + QPS chain

JJ and QPS elements can coexist in the same circuit, on the same or
different nodes.

```python
C = Capacitor(value=1, unit='GHz')
J = Junction(value=1,  unit='GHz', cap=C)
L = Inductor(value=1,  unit='GHz')
P = PhaseSlip(value=1, unit='GHz', ind=L)

# JJ on branch (0,1), QPS on branch (1,2)
circuit = Circuit([(0, 1, J), (1, 2, P)])

circuit.Hamiltonian_expression()
# H/ℏ (GHz) = + 4.0 (n_c1)² − 1.0 cos(v_1 ξφ) − 1.0 cos(u_1 ξq)

circuit.no_final_compact_flux    # 1
circuit.no_final_compact_charge  # 1
```

Both the compact flux (from J) and the compact charge (from P) appear
as independent nonlinear degrees of freedom.

---

## Example 6 — Multiple modes: 2 JJ in series

Each branch adds independent degrees of freedom. Two junctions in series
(with a closing capacitor) give two compact flux modes:

```python
C = Capacitor(value=1, unit='GHz')
J = Junction(value=1,  unit='GHz', cap=Capacitor(1,'GHz'))

circuit = Circuit([
    (0, 1, J),
    (1, 2, J),
    (0, 2, C),
])

circuit.quadratic_hamiltonian.shape   # (4, 4) — 2 modes × 2 variables each
circuit.no_final_compact_flux         # 2
circuit.vector_JJ.shape               # (4, 2) — one column per junction
```

---

## Reading the results

| Attribute | Type | Description |
|---|---|---|
| `circuit.quadratic_hamiltonian` | `ndarray (n×n)` | Quadratic part of H (harmonic + zero-point) |
| `circuit.extended_quantum_hamiltonian` | `ndarray (2k×2k)` | Diagonal: harmonic mode frequencies |
| `circuit.vector_JJ` | `ndarray (n×nJJ)` | Coupling vectors for each junction's cos(φ) |
| `circuit.vector_QPS` | `ndarray (n×nQPS)` | Coupling vectors for each phase-slip's cos(q) |
| `circuit.no_final_compact_flux` | `int` | Number of compact flux modes |
| `circuit.no_final_compact_charge` | `int` | Number of compact charge modes |
| `circuit.Hamiltonian_expression()` | `(prints)` | Symbolic Hamiltonian in GHz units |
| `circuit.FS_quadratic_hamiltonian_phiq` | `ndarray` | H in the flux-charge (φ,q) basis |
| `circuit.FS_quadratic_hamiltonian_an` | `ndarray` | H in the creation-annihilation (a†,a) basis |

---

## Common mistakes

**1. Forgetting the parallel cap/inductor:**
```python
Junction(value=10, unit='GHz')           # ← ValueError: cap= is required
PhaseSlip(value=5, unit='GHz')           # ← ValueError: ind= is required
```

**2. Wrong unit for Junction or PhaseSlip** (only frequency units are valid):
```python
Junction(value=1, unit='pF', cap=...)    # ← ValueError: unit not correct
```

**3. A circuit with no closed loop:**
```python
Circuit([(0, 1, Capacitor(1,'pF'))])     # ← Kirchhoff error: no return path
```
Every circuit needs at least one closed current path.

**4. Reusing the same element object on multiple branches:**
```python
J = Junction(1, 'GHz', cap=Capacitor(1,'GHz'))
Circuit([(0, 1, J), (0, 1, J)])   # ← same object twice — create two instances
```
Always instantiate a fresh element for each branch.

---

## Full working example (copy-paste ready)

```python
from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip
import numpy as np

# Fluxonium: JJ shunted by a superinductor
C_J = Capacitor(value=5,   unit='fF')
J   = Junction(value=8.0,  unit='GHz', cap=C_J)
L   = Inductor(value=400,  unit='nH')

circuit = Circuit([(0, 1, J), (0, 1, L)])

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

Quantum Hamiltonian:
H/ℏ (GHz) = + 31.6 [(ϕ_e1)² + (n_e1)²]  - 8.000 cos(v_1 ξφ)

JJ coupling vectors v (flux space):
v_1 = [-0.175  0.   ]
```
