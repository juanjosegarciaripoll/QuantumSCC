"""
Unit tests for QuantumSCC.core.quantization (Quantization class).

Tests exercise the class API directly with a minimal anonymous input.
No Circuit objects, no named architectures.  Each test asserts one
mathematical invariant of the Hamiltonian construction.

The input includes one JJ and one QPS element so that all six invariants
(including the coupling-vector checks) are exercised non-trivially.
"""

import numpy as np

from QuantumSCC.core.elements import Capacitor, Inductor, Junction, PhaseSlip
from QuantumSCC.core.geometry import Geometry
from QuantumSCC.core.quantization import Quantization
from QuantumSCC.core.topology import Topology

# Minimal input with one JJ and one QPS so every invariant is non-trivial.
# Anonymous — no circuit architecture is implied.
_T = Topology([
    (0, 1, Junction(1, 'GHz')),
    (0, 1, Capacitor(1, 'GHz')),
    (1, 2, PhaseSlip(1, 'GHz')),
    (1, 2, Inductor(1, 'GHz')),
])
_G = Geometry(_T)
_Q = Quantization(_T, _G)


def test_hamiltonian_symmetric():
    """Quadratic Hamiltonian must be symmetric: H = Hᵀ."""
    H = _Q.quadratic_hamiltonian
    assert np.allclose(H, H.T)


def test_hamiltonian_real():
    """Quadratic Hamiltonian must be real (no imaginary part)."""
    assert np.allclose(_Q.quadratic_hamiltonian.imag, 0)


def test_compact_counts_bounded():
    """Final compact counts cannot exceed generating element counts."""
    assert _G.no_final_compact_flux   <= _T.no_JJ
    assert _G.no_final_compact_charge <= _T.no_QPS


def test_jj_vector_nonzero():
    """vector_JJ must have a non-zero component in the dynamical sector."""
    assert not np.allclose(_Q.vector_JJ, 0)


def test_qps_vector_nonzero():
    """vector_QPS must have a non-zero component in the dynamical sector."""
    assert not np.allclose(_Q.vector_QPS, 0)


def test_harmonic_frequencies_nonnegative():
    """All harmonic-mode frequencies (diagonal of extended_quantum_hamiltonian) must be ≥ 0."""
    H = _Q.extended_quantum_hamiltonian
    if H.shape[0] == 0:
        return
    assert all(H[i, i].real >= -1e-10 for i in range(H.shape[0] // 2))
