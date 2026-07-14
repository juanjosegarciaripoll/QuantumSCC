# Extracting the model and building your own solver

QuantumSCC derives a circuit Hamiltonian and can *print* it, but for building
your own solver you want the model as **data**, not text. `Circuit.model()`
returns a single [`CircuitModel`](../src/QuantumSCC/model.py) dataclass that
contains everything needed to reconstruct the Hamiltonian — with no dependence
on the internal index conventions of the pipeline.

```python
from QuantumSCC import Circuit, Junction, Capacitor

circuit = Circuit([(0, 1, Junction(10, 'GHz')), (0, 1, Capacitor(0.2, 'GHz'))])
model = circuit.model()
print(model)
# CircuitModel(n_modes=1, compact_flux=1, compact_charge=0, extended=0,
#              n_JJ=1, n_QPS=0, units='GHz')
```

## 1. The model

The model is expressed in the **phiq** (ready-to-quantise) basis as

```
H/ℏ = ½ φᵀ K_flux φ  +  ½ nᵀ K_charge n
      − Σ_j E_J[j] · cos( Σ_m v_J[j, m] · φ_m )
      − Σ_k E_P[k] · cos( Σ_m v_P[k, m] · n_m )
```

For each of the `N = n_compact_flux + n_compact_charge + n_extended` modes
there is a flux operator `φ_m` and a conjugate charge operator `n_m`
(`[φ_m, n_m] = i`). Energies are in `units` (GHz), with `n = Q/2e` and
`φ = 2π Φ/Φ₀`.

## 2. The dataclass

| field | shape | meaning |
|---|---|---|
| `n_compact_flux` | int | number of Josephson modes `(φ_c, n_c)` |
| `n_compact_charge` | int | number of phase-slip modes `(ψ_c, q_c)` |
| `n_extended` | int | number of oscillator modes `(φ_e, n_e)` |
| `K_flux` | `(N, N)` | quadratic flux matrix, `H ⊃ ½ φᵀ K_flux φ` |
| `K_charge` | `(N, N)` | quadratic charge matrix, `H ⊃ ½ nᵀ K_charge n` |
| `E_J`, `v_J` | `(n_JJ,)`, `(n_JJ, N)` | junction amplitudes & **flux** coupling vectors |
| `E_P`, `v_P` | `(n_QPS,)`, `(n_QPS, N)` | phase-slip amplitudes & **charge** coupling vectors |

The **mode order is fixed**: `[compact_flux | compact_charge | extended]`, so a
mode index alone fixes its sector and Hilbert space. Helpers: `model.n_modes`,
`model.sector(m)`, `model.mode_sectors`, `model.mode_label(m)`.

## 3. Sectors → Hilbert spaces

| sector | pair | well-defined (linear) operator | periodic operator | suggested basis |
|---|---|---|---|---|
| `compact_flux` (JJ) | `(φ_c, n_c)` | `n_c` — integer charge | `φ_c` (enters only via `cos`) | charge basis `|n⟩`, `n ∈ [−n_cut, n_cut]` |
| `compact_charge` (QPS) | `(ψ_c, q_c)` | `ψ_c` — integer flux | `q_c` (enters only via `cos`) | dual charge basis `|ψ⟩` |
| `extended` (oscillator) | `(φ_e, n_e)` | both `φ_e`, `n_e` | — | Fock basis |

Two facts make reconstruction clean:

1. **Duality.** In the QPS sector the roles swap: the *flux* `ψ_c` is the
   integer/well-defined operator and the *charge* `q_c` is the periodic one
   (mirror image of the JJ sector).
2. **Invariant.** The periodic operators (`φ_c`, `q_c`) appear **only inside
   cosines**. Consequently `K_flux[m, m] = 0` for every `compact_flux` mode and
   `K_charge[m, m] = 0` for every `compact_charge` mode, and the quadratic part
   never references a periodic operator. Capacitive/inductive couplings always
   act between well-defined operators (`n_c`, `ψ_c`, `φ_e`, `n_e`).

### phiq structure you can exploit

- **Extended modes are isotropic normal modes**: `K_flux[m, m] == K_charge[m, m]`
  equals the mode frequency `ω`, so `½ K_flux φ² + ½ K_charge n² = ω(a†a + ½)`.
- **`K_flux`/`K_charge` are mostly diagonal.** Off-diagonal entries are exactly
  the residual cross-sector couplings (e.g. a transmon capacitively coupled to a
  resonator shows up as a single `K_charge[i, j]`).
- **Cosine coupling vectors are minimal** — each `v_J[j]` touches only the modes
  the junction actually phase-couples to. Inter-mode JJ/QPS couplings are carried
  *here*, not in the quadratic matrices.

## 4. Reference reconstruction (scipy.sparse)

The recipe: pick a Hilbert space per mode from its sector, build per-mode flux
and charge operators, then assemble the quadratic part from `K_flux`/`K_charge`
and the cosines from the coupling vectors. Because different modes live on
different tensor factors and commute, a cosine factorises:
`exp(i Σ_m c_m φ_m) = ⊗_m exp(i c_m φ_m)`.

```python
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm as dense_expm


def annihilation(d):
    return sp.diags(np.sqrt(np.arange(1, d)), 1, format="csc")


def reconstruct(model, charge_cut=12, fock_dim=12):
    """Build a sparse Hamiltonian (in GHz) from a CircuitModel."""
    secs = model.mode_sectors
    dims = [fock_dim if s == "extended" else 2 * charge_cut + 1 for s in secs]
    N, dim = model.n_modes, int(np.prod([fock_dim if s == "extended"
                                         else 2 * charge_cut + 1 for s in secs]))

    def embed(op, m):                       # place a single-mode op on the full space
        out = sp.identity(1, format="csc", dtype=complex)
        for k in range(N):
            out = sp.kron(out, op if k == m else sp.identity(dims[k]), format="csc")
        return out

    Phi, Nop = [None] * N, [None] * N       # linear operators (None where periodic)
    expF, expN = [None] * N, [None] * N     # coeff -> exp(i*coeff*coordinate)

    for m, s in enumerate(secs):
        d = dims[m]
        if s == "extended":
            a = annihilation(d); ad = a.conj().T
            phi, n = (a + ad) / np.sqrt(2), 1j * (ad - a) / np.sqrt(2)
            Phi[m], Nop[m] = phi, n
            expF[m] = lambda c, o=phi: sp.csc_matrix(dense_expm(1j * c * o.toarray()))
            expN[m] = lambda c, o=n:   sp.csc_matrix(dense_expm(1j * c * o.toarray()))
        elif s == "compact_flux":                       # |n⟩ charge basis
            vals = np.arange(-charge_cut, charge_cut + 1)
            Nop[m] = sp.diags(vals, 0, format="csc", dtype=float)        # integer n_c
            shift = sp.diags(np.ones(d - 1), -1, format="csc")          # e^{iφ}: n→n+1
            expF[m] = lambda c, S=shift: (S ** int(round(c)) if c >= 0
                                          else (S.conj().T) ** int(round(-c)))
            expN[m] = lambda c, v=vals: sp.diags(np.exp(1j * c * v), 0, format="csc")
        else:                                           # compact_charge: |ψ⟩ dual basis
            vals = np.arange(-charge_cut, charge_cut + 1)
            Phi[m] = sp.diags(vals, 0, format="csc", dtype=float)        # integer ψ_c
            shift = sp.diags(np.ones(d - 1), -1, format="csc")          # e^{iq}: ψ→ψ+1
            expN[m] = lambda c, S=shift: (S ** int(round(c)) if c >= 0
                                          else (S.conj().T) ** int(round(-c)))
            expF[m] = lambda c, v=vals: sp.diags(np.exp(1j * c * v), 0, format="csc")

    H = sp.csc_matrix((dim, dim), dtype=complex)

    # quadratic:  ½ Σ K_flux φ φ  +  ½ Σ K_charge n n
    for Kmat, ops in ((model.K_flux, Phi), (model.K_charge, Nop)):
        for a in range(N):
            for b in range(N):
                if abs(Kmat[a, b]) < 1e-12:
                    continue
                assert ops[a] is not None and ops[b] is not None, \
                    "quadratic references a periodic operator (invariant violated)"
                H = H + 0.5 * Kmat[a, b] * (embed(ops[a], a) @ embed(ops[b], b))

    # cosines (JJ over flux coords, QPS over charge coords)
    for E, V, expo in ((model.E_J, model.v_J, expF), (model.E_P, model.v_P, expN)):
        for t in range(len(E)):
            P = sp.identity(dim, format="csc", dtype=complex)
            for m in range(N):
                if abs(V[t, m]) >= 1e-12:
                    P = P @ embed(expo[m](V[t, m]), m)
            H = H - E[t] * 0.5 * (P + P.conj().T)

    return H
```

### Validation

```python
from scipy.sparse.linalg import eigsh
from QuantumSCC import Circuit, Junction, Capacitor, Inductor

def levels(H, k=4):
    return np.sort(eigsh(H, k=k, which="SA", return_eigenvectors=False).real)

# Transmon:  ω ≈ √(8 E_C E_J) − anharmonicity,  E_C = K_charge/8 = 0.05, E_J = 10
m = Circuit([(0,1,Junction(10,'GHz')), (0,1,Capacitor(0.2,'GHz'))]).model()
e = levels(reconstruct(m));  print(e[1]-e[0])      # ≈ 1.95 GHz

# LC oscillator: equally spaced by ω = 2 GHz
m = Circuit([(0,1,Inductor(2,'GHz')), (0,1,Capacitor(0.5,'GHz'))]).model()
e = levels(reconstruct(m));  print(np.diff(e))     # ≈ [2, 2, 2]
```

Observed: transmon 0→1 gap ≈ `1.9487` GHz (plasma frequency minus
anharmonicity), LC spacing exactly `2` GHz.

## 5. Notes for production solvers

- **Truncation.** `charge_cut`/`fock_dim` must be large enough; converge by
  increasing them until low-lying levels stabilise. Extended-mode frequencies
  (`K_flux[m, m]`) give a good initial Fock cutoff.
- **Cosines on oscillators.** `exp(i c φ_e)` is a displacement operator; the
  dense `expm` above is fine for moderate `fock_dim`. For large cutoffs use a
  sparse Krylov `expm` or build displacement operators directly.
- **Sparsity.** Inspect `model.K_flux`/`model.K_charge`: nonzero off-diagonals
  are your coupling graph. Pass `circuit.model(tol=0)` to keep raw
  (un-sparsified) values.
- **Basis source.** The model uses the phiq basis. The raw Darboux basis
  (`circuit.quadratic_hamiltonian`, `circuit.vector_JJ`, `circuit.vector_QPS`)
  is also available if you prefer to run your own symplectic diagonalisation.
