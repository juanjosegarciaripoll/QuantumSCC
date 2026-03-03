"""
Unit tests for QuantumSCC.core.topology (Topology class).

Tests exercise the class API directly with a minimal anonymous input.
No Circuit objects, no named architectures.  Each test asserts one
mathematical invariant of the Topology construction.
"""

import numpy as np
import os
import sys

current_dir  = os.path.dirname(os.path.abspath(__file__))
package_dir  = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC.core.elements import Capacitor, Inductor, Junction, PhaseSlip
from QuantumSCC.core.topology import Topology

# Minimal valid input: 2 nodes, 1 inductor + 1 capacitor.
# Anonymous — no circuit architecture is implied.
_T = Topology([
    (0, 1, Inductor(1, 'GHz')),
    (0, 1, Capacitor(1, 'GHz')),
])


def test_kirchhoff():
    """F · K = 0 (fundamental Kirchhoff constraint)."""
    assert np.allclose(_T.F @ _T.K, 0)


def test_rank_nullity():
    """rank(F) + cols(K) = 2 · no_elements (rank-nullity theorem on 2B space)."""
    assert np.linalg.matrix_rank(_T.F) + _T.K.shape[1] == 2 * _T.no_elements


def test_element_counts_consistent():
    """Sum of typed counts must equal no_elements."""
    total = _T.no_JJ + _T.no_Capacitors + _T.no_QPS + _T.no_Inductors
    assert total == _T.no_elements


def test_element_ordering():
    """Elements list must follow the fixed order [JJ | Cap | QPS | Ind]."""
    _group = {Junction: 0, Capacitor: 1, PhaseSlip: 2, Inductor: 3}
    last = -1
    for _, _, el in _T.elements:
        g = _group[type(el)]
        assert g >= last
        last = g


def test_compact_flux_bounded_by_jj():
    """no_reduced_compact_flux cannot exceed no_JJ."""
    assert _T.no_reduced_compact_flux <= _T.no_JJ


def test_compact_charge_bounded_by_qps():
    """no_reduced_compact_charge cannot exceed no_QPS."""
    assert _T.no_reduced_compact_charge <= _T.no_QPS
