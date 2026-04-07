"""
test_stress.py — 100-circuit stress test battery

Verifies the QuantumSCC algorithm through:
  A. Structural invariants (every circuit): FK=0, integer kernel, V^T Ω V = J
  B. Analytical Hamiltonian values: frequencies, H_quad entries, mode counts
  C. Independent LC nodal computation cross-check (fully independent of pipeline)
  D. Parameter scaling consistency
  E. JJ ↔ QPS duality

Each test verifies the output against a KNOWN value derived independently
from the algorithm.  For LC circuits, frequencies are cross-checked against
the generalized eigenvalue problem K_node v = ω² C_node v built from
physical capacitances/inductances — a computation that shares NO code with
the QuantumSCC pipeline.
"""

import sys
import os
import pytest
import numpy as np
from scipy.linalg import eigh

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip


# ── Shared invariant checker ─────────────────────────────────────────────────

def assert_invariants(circ):
    """Verify all structural invariants for a circuit."""
    topo = circ.topo
    geom = circ.geom
    ne = topo.no_elements
    nCF = topo.no_reduced_compact_flux
    nCC = topo.no_reduced_compact_charge

    # 1. Kernel property: F @ K = 0
    assert np.allclose(topo.F @ topo.K, 0), "F @ K != 0"

    # 2. Rank-nullity theorem
    rank_F = np.linalg.matrix_rank(topo.F)
    assert rank_F + topo.K.shape[1] == 2 * ne, \
        f"rank-nullity: {rank_F} + {topo.K.shape[1]} != {2 * ne}"

    # 3. Compact flux columns have integer entries
    if nCF > 0:
        Kloop_compact = topo.K[:ne, :nCF]
        assert np.allclose(Kloop_compact, np.round(Kloop_compact)), \
            "Compact flux K columns are not integer"

    # 4. Compact charge columns have integer entries
    if nCC > 0:
        n_flux_vars = topo.Fcut.shape[0]
        Kcut_compact = topo.K[ne:, n_flux_vars:n_flux_vars + nCC]
        assert np.allclose(Kcut_compact, np.round(Kcut_compact)), \
            "Compact charge K columns are not integer"

    # 5. V^T @ omega_ns @ V = J (canonical symplectic form)
    omega_ns = topo.K.T @ geom.omega_2B @ topo.K
    J_check = geom.V.T @ omega_ns @ geom.V
    n_indep = geom.no_independent_variables
    nF = n_indep // 2
    J_expected = np.zeros_like(J_check)
    J_expected[:nF, nF:2 * nF] = np.eye(nF)
    J_expected[nF:2 * nF, :nF] = -np.eye(nF)
    assert np.allclose(J_check[:2 * nF, :2 * nF],
                       J_expected[:2 * nF, :2 * nF], atol=1e-10), \
        "V^T @ omega_ns @ V != J"

    # 6. Quadratic Hamiltonian is symmetric
    H = circ.quadratic_hamiltonian
    assert np.allclose(H, H.T), "H_quad not symmetric"

    # 7. Extended Hamiltonian eigenvalues are non-negative
    H_ext = circ.extended_quantum_hamiltonian.real
    n_ext = H_ext.shape[0] // 2
    if n_ext > 0:
        freqs = [H_ext[i, i] for i in range(n_ext)]
        assert all(f > -1e-10 for f in freqs), f"Negative freq: {freqs}"


# ── Independent LC frequency computation ─────────────────────────────────────

def independent_lc_freqs(edges):
    """
    Compute LC mode frequencies using nodal admittance matrices.

    Builds C_node and K_node (= 1/L_node) from physical element values,
    grounds node 0, and solves the generalized eigenvalue problem
    K_red v = ω² C_red v.  Returns ω / 1e9 (code's GHz convention).

    This computation shares NO code with the QuantumSCC pipeline.
    Only valid for circuits with Capacitors and Inductors (no JJ, no QPS).
    """
    nodes = set()
    for a, b, _ in edges:
        nodes.add(a)
        nodes.add(b)
    nodes = sorted(nodes)
    n = len(nodes)
    idx = {nd: i for i, nd in enumerate(nodes)}

    C_node = np.zeros((n, n))
    K_node = np.zeros((n, n))

    for a, b, elem in edges:
        i, j = idx[a], idx[b]
        if isinstance(elem, Capacitor):
            c = elem.value()          # Farads
            C_node[i, i] += c
            C_node[j, j] += c
            C_node[i, j] -= c
            C_node[j, i] -= c
        elif isinstance(elem, Inductor):
            k = 1.0 / elem.value()    # 1/Henry
            K_node[i, i] += k
            K_node[j, j] += k
            K_node[i, j] -= k
            K_node[j, i] -= k

    # Ground node 0: remove row/col 0
    C_red = C_node[1:, 1:]
    K_red = K_node[1:, 1:]

    if C_red.shape[0] == 0:
        return np.array([])

    # Generalized eigenvalue: K v = ω² C v
    eigvals = eigh(K_red, C_red, eigvals_only=True)
    eigvals = eigvals[eigvals > 1e-6]
    return np.sort(np.sqrt(eigvals)) / 1e9


# ── Analytical helper ─────────────────────────────────────────────────────────

def lc_freq(E_C, E_L):
    """Analytical LC frequency: ω = 2√(E_C·E_L) in code GHz."""
    return 2.0 * np.sqrt(E_C * E_L)


# ── Element factories ─────────────────────────────────────────────────────────

def _J(E_J=1):
    return Junction(E_J, 'GHz')

def _P(E_P=1):
    return PhaseSlip(E_P, 'GHz')

def _C(E_C=1):
    return Capacitor(E_C, 'GHz')

def _L(E_L=1):
    return Inductor(E_L, 'GHz')


# ============================================================================
# GROUP A: LC oscillator — ω = 2√(E_C·E_L) + independent cross-check
#          20 tests
# ============================================================================

LC_PARAMS = [
    (1, 1), (2, 1), (1, 2), (5, 3), (0.1, 10),
    (10, 0.1), (3, 7), (0.5, 0.5), (100, 0.01), (4, 4),
    (2.5, 6), (8, 0.25), (1.5, 3.5), (0.3, 0.7), (20, 5),
    (7, 7), (0.05, 200), (50, 2), (1.1, 0.9), (3.14, 2.72),
]

@pytest.mark.parametrize("E_C,E_L", LC_PARAMS,
    ids=[f"EC{a}_EL{b}" for a, b in LC_PARAMS])
def test_lc_frequency(E_C, E_L):
    """LC oscillator: ω = 2√(E_C·E_L), 0 compact modes, 1 harmonic mode."""
    edges = [(0, 1, _L(E_L)), (0, 1, _C(E_C))]
    circ = Circuit(edges)
    assert_invariants(circ)

    omega = circ.extended_quantum_hamiltonian.real[0, 0]
    expected = lc_freq(E_C, E_L)
    assert np.allclose(omega, expected, rtol=1e-6), \
        f"code={omega:.6f}, analytical={expected:.6f}"
    assert circ.no_final_compact_flux == 0
    assert circ.no_final_compact_charge == 0
    assert circ.no_independent_variables == 2

    # Independent nodal cross-check
    indep = independent_lc_freqs(edges)
    assert np.allclose(omega, indep[0], rtol=1e-6), \
        f"code={omega:.6f}, independent={indep[0]:.6f}"


# ============================================================================
# GROUP B: Transmon — H_quad = diag(0, 2E_C), nCF=1
#          10 tests
# ============================================================================

TRANSMON_PARAMS = [
    (10, 1), (5, 0.5), (20, 2), (1, 1), (50, 0.1),
    (100, 10), (0.5, 0.5), (15, 3), (8, 0.8), (30, 1.5),
]

@pytest.mark.parametrize("E_J,E_C", TRANSMON_PARAMS,
    ids=[f"EJ{a}_EC{b}" for a, b in TRANSMON_PARAMS])
def test_transmon_hamiltonian(E_J, E_C):
    """Transmon: H[0,0]=0 (compact flux), H[1,1]=2E_C, nCF=1, nCC=0."""
    circ = Circuit([(0, 1, _J(E_J)), (0, 1, _C(E_C))])
    assert_invariants(circ)

    H = circ.quadratic_hamiltonian
    assert np.abs(H[0, 0]) < 1e-10, f"H[0,0]={H[0, 0]} (expected 0)"
    assert np.allclose(H[1, 1], 2 * E_C, rtol=1e-6), \
        f"H[1,1]={H[1, 1]:.6f}, expected {2 * E_C:.6f}"
    assert circ.no_final_compact_flux == 1
    assert circ.no_final_compact_charge == 0
    assert circ.vector_JJ.shape[1] == 1


# ============================================================================
# GROUP C: Fluxonium — ω = 2√(E_C·E_L), nCF=0
#          Inductor kills compact flux. Extended mode = JJ cap × ext inductor.
#          10 tests
# ============================================================================

FLUXONIUM_PARAMS = [
    (10, 1, 1), (20, 2, 0.5), (5, 0.5, 3), (15, 1, 2),
    (50, 5, 0.1), (8, 0.8, 0.8), (1, 1, 1), (30, 3, 1.5),
    (100, 10, 0.01), (3, 0.3, 7),
]

@pytest.mark.parametrize("E_J,E_C,E_L", FLUXONIUM_PARAMS,
    ids=[f"EJ{a}_EC{b}_EL{c}" for a, b, c in FLUXONIUM_PARAMS])
def test_fluxonium_frequency(E_J, E_C, E_L):
    """Fluxonium: inductor kills compact flux → ω = 2√(E_C·E_L)."""
    circ = Circuit([(0, 1, _J(E_J)), (0, 1, _C(E_C)), (0, 1, _L(E_L))])
    assert_invariants(circ)

    omega = circ.extended_quantum_hamiltonian.real[0, 0]
    expected = lc_freq(E_C, E_L)
    assert np.allclose(omega, expected, rtol=1e-6), \
        f"code={omega:.6f}, analytical={expected:.6f}"
    assert circ.no_final_compact_flux == 0
    assert circ.no_final_compact_charge == 0
    assert circ.vector_JJ.shape[1] == 1


# ============================================================================
# GROUP D: Dual transmon — H_quad = diag(2E_L, 0), nCC=1
#          10 tests
# ============================================================================

DUAL_TRANSMON_PARAMS = [
    (5, 1), (10, 2), (1, 1), (20, 0.5), (3, 3),
    (50, 0.1), (0.5, 0.5), (15, 5), (8, 0.8), (100, 10),
]

@pytest.mark.parametrize("E_P,E_L", DUAL_TRANSMON_PARAMS,
    ids=[f"EP{a}_EL{b}" for a, b in DUAL_TRANSMON_PARAMS])
def test_dual_transmon_hamiltonian(E_P, E_L):
    """Dual transmon: H[0,0]=2E_L, H[1,1]=0 (compact charge), nCC=1."""
    circ = Circuit([(0, 1, _P(E_P)), (0, 1, _L(E_L))])
    assert_invariants(circ)

    H = circ.quadratic_hamiltonian
    assert np.allclose(H[0, 0], 2 * E_L, rtol=1e-6), \
        f"H[0,0]={H[0, 0]:.6f}, expected {2 * E_L:.6f}"
    assert np.abs(H[1, 1]) < 1e-10, f"H[1,1]={H[1, 1]} (expected 0)"
    assert circ.no_final_compact_flux == 0
    assert circ.no_final_compact_charge == 1
    assert circ.vector_QPS.shape[1] == 1


# ============================================================================
# GROUP E: Dual fluxonium — ω = 2√(E_C·E_L), nCC=0
#          Capacitor kills compact charge. Extended mode = ext cap × QPS inductor.
#          10 tests
# ============================================================================

DUAL_FLUX_PARAMS = [
    (5, 1, 1), (10, 2, 0.5), (1, 3, 0.5), (20, 0.5, 2),
    (3, 3, 3), (50, 0.1, 5), (0.5, 0.5, 0.5), (15, 5, 1.5),
    (8, 0.8, 0.8), (100, 10, 0.01),
]

@pytest.mark.parametrize("E_P,E_L,E_C", DUAL_FLUX_PARAMS,
    ids=[f"EP{a}_EL{b}_EC{c}" for a, b, c in DUAL_FLUX_PARAMS])
def test_dual_fluxonium_frequency(E_P, E_L, E_C):
    """Dual fluxonium: capacitor kills compact charge → ω = 2√(E_C·E_L)."""
    circ = Circuit([(0, 1, _P(E_P)), (0, 1, _L(E_L)), (0, 1, _C(E_C))])
    assert_invariants(circ)

    omega = circ.extended_quantum_hamiltonian.real[0, 0]
    expected = lc_freq(E_C, E_L)
    assert np.allclose(omega, expected, rtol=1e-6), \
        f"code={omega:.6f}, analytical={expected:.6f}"
    assert circ.no_final_compact_flux == 0
    assert circ.no_final_compact_charge == 0
    assert circ.vector_QPS.shape[1] == 1


# ============================================================================
# GROUP F: Coupled LC — normal mode frequencies + independent cross-check
#          Two identical LC oscillators coupled by capacitor Cg.
#          ω_sym = 1/√(LC),  ω_anti = 1/√(L(C+2Cg))  (physical units)
#          5 tests
# ============================================================================

COUPLED_PARAMS = [
    (1.0, 1.0, 2.0),
    (2.0, 1.0, 1.0),
    (1.0, 2.0, 0.5),
    (5.0, 0.5, 3.0),
    (0.5, 5.0, 1.0),
]

@pytest.mark.parametrize("C_pF,L_nH,Cg_pF", COUPLED_PARAMS,
    ids=[f"C{a}_L{b}_Cg{c}" for a, b, c in COUPLED_PARAMS])
def test_coupled_lc_frequencies(C_pF, L_nH, Cg_pF):
    """Coupled LC: analytical normal mode frequencies + independent check."""
    C1 = Capacitor(C_pF, 'pF')
    C2 = Capacitor(C_pF, 'pF')
    Cg = Capacitor(Cg_pF, 'pF')
    L1 = Inductor(L_nH, 'nH')
    L2 = Inductor(L_nH, 'nH')
    edges = [(0, 1, L1), (1, 2, Cg), (2, 0, L2), (0, 1, C1), (2, 0, C2)]
    circ = Circuit(edges)
    assert_invariants(circ)

    # Analytical: physical units
    C_F = C_pF * 1e-12
    Cg_F = Cg_pF * 1e-12
    L_H = L_nH * 1e-9
    omega_sym = np.sqrt(1.0 / (L_H * C_F)) / 1e9
    omega_anti = np.sqrt(1.0 / (L_H * (C_F + 2 * Cg_F))) / 1e9

    H = circ.extended_quantum_hamiltonian.real
    freqs = sorted([H[i, i] for i in range(H.shape[0] // 2)])

    assert np.allclose(freqs[0], omega_anti, rtol=1e-4), \
        f"anti: code={freqs[0]:.4f}, theory={omega_anti:.4f}"
    assert np.allclose(freqs[1], omega_sym, rtol=1e-4), \
        f"sym: code={freqs[1]:.4f}, theory={omega_sym:.4f}"

    # Independent nodal cross-check
    indep = sorted(independent_lc_freqs(edges))
    assert np.allclose(freqs, indep, rtol=1e-4), \
        f"code={freqs}, independent={indep}"


# ============================================================================
# GROUP G: JJ ↔ QPS duality — structural correspondence
#          Transmon ↔ dual transmon: swapped compact modes, same quadratic energy
#          Fluxonium ↔ dual fluxonium: both lose compact, same ω
#          10 tests (5 transmon pairs + 5 fluxonium pairs)
# ============================================================================

DUALITY_PARAMS = [
    (1, 10), (0.5, 5), (2, 20), (3, 30), (5, 1),
]

@pytest.mark.parametrize("E_val,E_nonlin", DUALITY_PARAMS,
    ids=[f"E{a}_Enl{b}" for a, b in DUALITY_PARAMS])
def test_transmon_duality(E_val, E_nonlin):
    """Transmon ↔ dual transmon: swapped compact modes, dual H structure."""
    transmon = Circuit([(0, 1, _J(E_nonlin)), (0, 1, _C(E_val))])
    dual = Circuit([(0, 1, _P(E_nonlin)), (0, 1, _L(E_val))])
    assert_invariants(transmon)
    assert_invariants(dual)

    # Compact modes swapped
    assert transmon.no_final_compact_flux == 1
    assert transmon.no_final_compact_charge == 0
    assert dual.no_final_compact_flux == 0
    assert dual.no_final_compact_charge == 1

    # H structure swapped: transmon H=diag(0,2E_C) vs dual H=diag(2E_L,0)
    H_t = transmon.quadratic_hamiltonian
    H_d = dual.quadratic_hamiltonian
    assert np.abs(H_t[0, 0]) < 1e-10
    assert np.abs(H_d[1, 1]) < 1e-10
    assert np.allclose(H_t[1, 1], 2 * E_val, rtol=1e-6)
    assert np.allclose(H_d[0, 0], 2 * E_val, rtol=1e-6)


@pytest.mark.parametrize("E_val,E_nonlin", DUALITY_PARAMS,
    ids=[f"E{a}_Enl{b}" for a, b in DUALITY_PARAMS])
def test_fluxonium_duality(E_val, E_nonlin):
    """Fluxonium ↔ dual fluxonium: both 0 compact modes, same ω."""
    flux = Circuit([(0, 1, _J(E_nonlin)), (0, 1, _C(E_val)), (0, 1, _L(E_val))])
    dual = Circuit([(0, 1, _P(E_nonlin)), (0, 1, _L(E_val)), (0, 1, _C(E_val))])
    assert_invariants(flux)
    assert_invariants(dual)

    assert flux.no_final_compact_flux == 0
    assert flux.no_final_compact_charge == 0
    assert dual.no_final_compact_flux == 0
    assert dual.no_final_compact_charge == 0

    # Both have ω = 2√(E_val²) = 2·E_val
    expected = 2.0 * E_val
    w_flux = flux.extended_quantum_hamiltonian.real[0, 0]
    w_dual = dual.extended_quantum_hamiltonian.real[0, 0]
    assert np.allclose(w_flux, expected, rtol=1e-6)
    assert np.allclose(w_dual, expected, rtol=1e-6)


# ============================================================================
# GROUP H: Scaling consistency — ω ∝ √(E_C·E_L)
#          Scale E_C or E_L by factor s → ω scales by √s
#          10 tests (5 capacitance + 5 inductance)
# ============================================================================

SCALE_PARAMS = [
    (1, 1, 2), (1, 1, 4), (5, 3, 3), (0.5, 2, 0.5), (10, 1, 10),
]

@pytest.mark.parametrize("E_C,E_L,s", SCALE_PARAMS,
    ids=[f"EC{a}_EL{b}_x{c}" for a, b, c in SCALE_PARAMS])
def test_lc_scale_capacitance(E_C, E_L, s):
    """Scaling E_C by s → ω scales by √s."""
    c1 = Circuit([(0, 1, _L(E_L)), (0, 1, _C(E_C))])
    c2 = Circuit([(0, 1, _L(E_L)), (0, 1, _C(E_C * s))])
    w1 = c1.extended_quantum_hamiltonian.real[0, 0]
    w2 = c2.extended_quantum_hamiltonian.real[0, 0]
    assert np.allclose(w2 / w1, np.sqrt(s), rtol=1e-6)


@pytest.mark.parametrize("E_C,E_L,s", SCALE_PARAMS,
    ids=[f"EC{a}_EL{b}_x{c}" for a, b, c in SCALE_PARAMS])
def test_lc_scale_inductance(E_C, E_L, s):
    """Scaling E_L by s → ω scales by √s."""
    c1 = Circuit([(0, 1, _L(E_L)), (0, 1, _C(E_C))])
    c2 = Circuit([(0, 1, _L(E_L * s)), (0, 1, _C(E_C))])
    w1 = c1.extended_quantum_hamiltonian.real[0, 0]
    w2 = c2.extended_quantum_hamiltonian.real[0, 0]
    assert np.allclose(w2 / w1, np.sqrt(s), rtol=1e-6)


# ============================================================================
# GROUP I: Multi-node LC — independent nodal cross-check
#          Verifies multi-mode circuits against fully independent computation.
#          5 tests
# ============================================================================

MULTI_LC = [
    ("star_3arm",
     [(0, 1, _L(1)), (0, 1, _C(1)),
      (0, 2, _L(2)), (0, 2, _C(2)),
      (0, 3, _L(3)), (0, 3, _C(3))]),
    ("triangle",
     [(0, 1, _L(1)), (0, 1, _C(2)),
      (1, 2, _L(3)), (1, 2, _C(4)),
      (2, 0, _L(5)), (2, 0, _C(6))]),
    ("chain_3",
     [(0, 1, _L(1)), (0, 1, _C(2)),
      (1, 2, _L(3)), (1, 2, _C(4)),
      (2, 3, _L(5)), (2, 3, _C(6))]),
    ("star_4arm",
     [(0, 1, _L(1)), (0, 1, _C(1)),
      (0, 2, _L(2)), (0, 2, _C(2)),
      (0, 3, _L(3)), (0, 3, _C(3)),
      (0, 4, _L(4)), (0, 4, _C(4))]),
    ("2C_1L_parallel",
     [(0, 1, _L(3)), (0, 1, _C(1)), (0, 1, _C(2))]),
]

@pytest.mark.parametrize("name,edges", MULTI_LC, ids=[c[0] for c in MULTI_LC])
def test_multi_lc_independent(name, edges):
    """Multi-node LC: code frequencies match independent nodal computation."""
    circ = Circuit(edges)
    assert_invariants(circ)

    H = circ.extended_quantum_hamiltonian.real
    n_modes = H.shape[0] // 2
    code_freqs = sorted([H[i, i] for i in range(n_modes)])

    indep_freqs = sorted(independent_lc_freqs(edges))

    assert len(code_freqs) == len(indep_freqs), \
        f"{name}: {len(code_freqs)} code modes vs {len(indep_freqs)} independent"
    assert np.allclose(code_freqs, indep_freqs, rtol=1e-4), \
        f"{name}: code={code_freqs}, indep={indep_freqs}"


# ============================================================================
# GROUP J: Complex topologies — structural invariants + mode count
#          JJ/QPS combinations that stress the algorithm most.
#          Each JJ has an explicit parallel Capacitor, each QPS has an
#          explicit parallel Inductor (replicating old companion behavior).
#          15 tests
# ============================================================================

COMPLEX_CIRCUITS = [
    ("2JJ_parallel",
     [(0, 1, _J(5)), (0, 1, _C(2)), (0, 1, _J(3)), (0, 1, _C(1))]),
    ("3JJ_parallel",
     [(0, 1, _J(5)), (0, 1, _C(1)), (0, 1, _J(3)), (0, 1, _C(2)),
      (0, 1, _J(7)), (0, 1, _C(0.5))]),
    ("2JJ_series",
     [(0, 1, _J(5)), (0, 1, _C(1)), (1, 2, _J(3)), (1, 2, _C(2)),
      (0, 2, _C(0.5))]),
    ("2QPS_parallel",
     [(0, 1, _P(5)), (0, 1, _L(1)), (0, 1, _P(3)), (0, 1, _L(2))]),
    ("3QPS_parallel",
     [(0, 1, _P(5)), (0, 1, _L(1)), (0, 1, _P(3)), (0, 1, _L(2)),
      (0, 1, _P(7)), (0, 1, _L(0.5))]),
    ("2QPS_series",
     [(0, 1, _P(5)), (0, 1, _L(1)), (1, 2, _P(3)), (1, 2, _L(2)),
      (0, 2, _L(0.5))]),
    ("JJ_QPS_same",
     [(0, 1, _J(5)), (0, 1, _C(1)), (0, 1, _P(3)), (0, 1, _L(2))]),
    ("JJ_QPS_chain",
     [(0, 1, _J(5)), (0, 1, _C(1)), (1, 2, _P(3)), (1, 2, _L(2))]),
    ("2JJ_QPS_shared",
     [(0, 1, _J(5)), (0, 1, _C(1)), (1, 2, _J(3)), (1, 2, _C(2)),
      (0, 1, _P(7)), (0, 1, _L(0.5))]),
    ("JJ_QPS_JJ_chain",
     [(0, 1, _J(5)), (0, 1, _C(1)), (1, 2, _P(3)), (1, 2, _L(2)),
      (2, 3, _J(7)), (2, 3, _C(0.5))]),
    ("JJ_JJ_QPS_ring",
     [(0, 1, _J(5)), (0, 1, _C(1)), (1, 2, _J(3)), (1, 2, _C(2)),
      (2, 0, _P(7)), (2, 0, _L(0.5))]),
    ("QPS_QPS_JJ_ring",
     [(0, 1, _P(5)), (0, 1, _L(1)), (1, 2, _P(3)), (1, 2, _L(2)),
      (2, 0, _J(7)), (2, 0, _C(0.5))]),
    ("LC_JJ_coupled",
     [(0, 1, _L(1)), (0, 1, _C(1)), (1, 2, _J(5)), (1, 2, _C(1))]),
    ("LC_QPS_coupled",
     [(0, 1, _L(1)), (0, 1, _C(1)), (1, 2, _P(5)), (1, 2, _L(1))]),
    ("JJ_QPS_JJ_QPS_chain",
     [(0, 1, _J(5)), (0, 1, _C(1)), (1, 2, _P(3)), (1, 2, _L(2)),
      (2, 3, _J(7)), (2, 3, _C(0.5)), (3, 4, _P(2)), (3, 4, _L(3))]),
]

@pytest.mark.parametrize("name,edges", COMPLEX_CIRCUITS,
    ids=[c[0] for c in COMPLEX_CIRCUITS])
def test_complex_invariants(name, edges):
    """Complex JJ/QPS topologies: all structural invariants hold."""
    circ = Circuit(edges)
    assert_invariants(circ)

    # Additional: extended mode frequencies are finite and positive
    H_ext = circ.extended_quantum_hamiltonian.real
    n_ext = H_ext.shape[0] // 2
    for i in range(n_ext):
        assert H_ext[i, i] > 0, f"{name}: mode {i} has ω={H_ext[i, i]}"
        assert np.isfinite(H_ext[i, i]), f"{name}: mode {i} is not finite"


# ── GROUP J.2: Parallel QPS duality — doubly-discrete gauge fix ──────────────

@pytest.mark.parametrize("n_qps", [2, 3])
def test_parallel_qps_duality(n_qps):
    """N parallel QPS: all vectors identical, H_quad = 2·Σ E_Li, dual JJ vectors also identical."""
    E_P = [5, 3, 7][:n_qps]
    E_L = [1, 2, 0.5][:n_qps]

    # QPS circuit
    qps_edges = ([(0, 1, _P(E_P[i])) for i in range(n_qps)]
                 + [(0, 1, _L(E_L[i])) for i in range(n_qps)])
    circ_qps = Circuit(qps_edges)

    # All QPS vectors non-zero and identical
    for col in range(n_qps):
        assert not np.allclose(circ_qps.vector_QPS[:, col], 0), \
            f"QPS {col} has zero coupling vector"
    for col in range(1, n_qps):
        np.testing.assert_allclose(
            circ_qps.vector_QPS[:, col], circ_qps.vector_QPS[:, 0],
            atol=1e-12, err_msg=f"QPS {col} vector differs from QPS 0")

    # Effective inductance: H_quad non-zero diagonal = 2·Σ E_Li
    diag = np.diag(circ_qps.quadratic_hamiltonian)
    nonzero = diag[np.abs(diag) > 1e-10]
    np.testing.assert_allclose(nonzero[0], 2.0 * sum(E_L), atol=1e-10,
                               err_msg=f"Expected 2·Σ E_L = {2*sum(E_L)}")

    # Dual JJ circuit: all JJ vectors should also be identical
    jj_edges = ([(0, 1, _J(E_P[i])) for i in range(n_qps)]
                + [(0, 1, _C(E_L[i])) for i in range(n_qps)])
    circ_jj = Circuit(jj_edges)
    for col in range(1, n_qps):
        np.testing.assert_allclose(
            circ_jj.vector_JJ[:, col], circ_jj.vector_JJ[:, 0],
            atol=1e-12, err_msg=f"JJ {col} vector differs from JJ 0")

    # Dual effective capacitance: 2·(1/Σ(1/E_Ci)) = harmonic combination
    diag_jj = np.diag(circ_jj.quadratic_hamiltonian)
    nonzero_jj = diag_jj[np.abs(diag_jj) > 1e-10]
    E_C_eff = 1.0 / sum(1.0 / e for e in E_L)
    np.testing.assert_allclose(nonzero_jj[0], 2.0 * E_C_eff, atol=1e-10,
                               err_msg=f"Expected 2·E_C_eff = {2*E_C_eff}")


# ============================================================================
# GROUP K: Physical units — verify pF/nH give same result as GHz
#          5 tests
# ============================================================================

PHYS_PARAMS = [
    (1.0, 1.0),    # 1 pF, 1 nH
    (0.5, 2.0),
    (10.0, 0.1),
    (0.1, 10.0),
    (3.0, 3.0),
]

@pytest.mark.parametrize("C_pF,L_nH", PHYS_PARAMS,
    ids=[f"C{a}pF_L{b}nH" for a, b in PHYS_PARAMS])
def test_physical_units_lc(C_pF, L_nH):
    """LC in pF/nH matches analytical ω = 1/√(LC) / 1e9."""
    C = Capacitor(C_pF, 'pF')
    L = Inductor(L_nH, 'nH')
    edges = [(0, 1, L), (0, 1, C)]
    circ = Circuit(edges)
    assert_invariants(circ)

    omega_code = circ.extended_quantum_hamiltonian.real[0, 0]
    omega_theory = np.sqrt(1.0 / (C_pF * 1e-12 * L_nH * 1e-9)) / 1e9

    assert np.allclose(omega_code, omega_theory, rtol=1e-6), \
        f"code={omega_code:.6f}, theory={omega_theory:.6f}"

    # Independent cross-check
    indep = independent_lc_freqs(edges)
    assert np.allclose(omega_code, indep[0], rtol=1e-6)


# ============================================================================
# TOTAL: 20 + 10 + 10 + 10 + 10 + 5 + 10 + 10 + 5 + 15 + 5 = 110 tests
# ============================================================================
