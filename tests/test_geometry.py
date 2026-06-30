"""
Unit tests for QuantumSCC.core.geometry (Geometry class).

Tests exercise the class API directly with a minimal anonymous input.
No Circuit objects, no named architectures.  Each test asserts one
mathematical invariant of the symplectic form / Darboux reduction.
"""

import numpy as np

from QuantumSCC.core.elements import Capacitor, Inductor
from QuantumSCC.core.topology import Topology
from QuantumSCC.core.geometry import Geometry

# Minimal valid input: 2 nodes, 1 inductor + 1 capacitor.
# Anonymous — no circuit architecture is implied.
_T = Topology([
    (0, 1, Inductor(1, 'GHz')),
    (0, 1, Capacitor(1, 'GHz')),
])
_G = Geometry(_T)


def test_omega_2b_antisymmetric():
    """omega_2B must be antisymmetric: Ω = −Ωᵀ."""
    O = _G.omega_2B
    assert np.allclose(O, -O.T)


def test_symplectic_form_antisymmetric():
    """omega_symplectic must be antisymmetric."""
    O = _G.omega_symplectic
    assert np.allclose(O, -O.T)


def test_symplectic_rank():
    """rank(omega_symplectic) == no_independent_variables (Darboux is non-degenerate)."""
    O = _G.omega_symplectic
    assert np.linalg.matrix_rank(O) == _G.no_independent_variables


def test_V_square():
    """Basis-change matrix V must be square."""
    assert _G.V.shape[0] == _G.V.shape[1]


def test_compact_flux_refines_reduced():
    """no_final_compact_flux ≤ no_reduced_compact_flux."""
    assert _G.no_final_compact_flux <= _T.no_reduced_compact_flux


def test_compact_charge_refines_reduced():
    """no_final_compact_charge ≤ no_reduced_compact_charge."""
    assert _G.no_final_compact_charge <= _T.no_reduced_compact_charge
