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


# ── PRX Eq. 42 integer kernel verification ─────────────────────────────────
# The compact columns of K must have integer entries (0, ±1, ±2, ...).
# This is the defining property of the integer_null_space algorithm.
import pytest

_CIRCUITS = {
    'LC': [(0, 1, Inductor(1, 'GHz')), (0, 1, Capacitor(1, 'GHz'))],
    'JJ': [(0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz'))],
    'JJ+L': [(0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz')),
             (0, 1, Inductor(1, 'GHz'))],
    'QPS': [(0, 1, PhaseSlip(1, 'GHz')), (0, 1, Inductor(1, 'GHz'))],
    'JJ||QPS': [(0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz')),
                (0, 1, PhaseSlip(1, 'GHz')), (0, 1, Inductor(1, 'GHz'))],
    'JJ-QPS': [(0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz')),
               (1, 2, PhaseSlip(1, 'GHz')), (1, 2, Inductor(1, 'GHz'))],
    '2JJ': [(0, 1, Junction(1, 'GHz')), (0, 1, Capacitor(1, 'GHz')),
            (1, 2, Junction(1, 'GHz')), (1, 2, Capacitor(1, 'GHz')),
            (0, 2, Capacitor(1, 'GHz'))],
}


@pytest.mark.parametrize('name', _CIRCUITS.keys())
def test_FK_zero(name):
    """PRX Eq. (7): F · K = 0 for all circuits."""
    topo = Topology(_CIRCUITS[name])
    assert np.allclose(topo.F @ topo.K, 0)


@pytest.mark.parametrize('name', _CIRCUITS.keys())
def test_compact_columns_integer(name):
    """PRX Eq. (8,11): Compact kernel columns must have integer entries."""
    topo = Topology(_CIRCUITS[name])
    ne = topo.no_elements
    nCF = topo.no_reduced_compact_flux
    nCC = topo.no_reduced_compact_charge
    n_flux_vars = topo.Fcut.shape[0]  # Kloop columns

    # Compact flux columns (first nCF of Kloop)
    if nCF > 0:
        Kloop_compact = topo.K[:ne, :nCF]
        assert np.allclose(Kloop_compact, np.round(Kloop_compact)), \
            f"{name}: compact flux columns are not integer"

    # Compact charge columns (first nCC of Kcut)
    if nCC > 0:
        Kcut_compact = topo.K[ne:, n_flux_vars:n_flux_vars + nCC]
        assert np.allclose(Kcut_compact, np.round(Kcut_compact)), \
            f"{name}: compact charge columns are not integer"


@pytest.mark.parametrize('name', _CIRCUITS.keys())
def test_rank_nullity_all(name):
    """rank(F) + cols(K) = 2·N_e (rank-nullity theorem)."""
    topo = Topology(_CIRCUITS[name])
    assert np.linalg.matrix_rank(topo.F) + topo.K.shape[1] == 2 * topo.no_elements


# ── PRX Eq. 42 two-topology decomposition tests ─────────────────────────

@pytest.mark.parametrize('name', _CIRCUITS.keys())
def test_Dloop_Eloop_reconstruct_Floop(name):
    """[D_loop | E_loop] must reconstruct Floop (Eq. 42 flux topology)."""
    topo = Topology(_CIRCUITS[name])
    reconstructed = np.hstack([topo.D_loop, topo.E_loop])
    assert np.allclose(reconstructed, topo.Floop), \
        f"{name}: [D_loop | E_loop] != Floop"


@pytest.mark.parametrize('name', _CIRCUITS.keys())
def test_Ecut_Dcut_reconstruct_Fcut(name):
    """[E_cut | D_cut] must reconstruct Fcut (Eq. 42 charge topology)."""
    topo = Topology(_CIRCUITS[name])
    reconstructed = np.hstack([topo.E_cut, topo.D_cut])
    assert np.allclose(reconstructed, topo.Fcut), \
        f"{name}: [E_cut | D_cut] != Fcut"


@pytest.mark.parametrize('name', _CIRCUITS.keys())
def test_Dloop_column_count(name):
    """D_loop must have no_JJ + no_Capacitors columns (two-island count)."""
    topo = Topology(_CIRCUITS[name])
    assert topo.D_loop.shape[1] == topo.no_JJ + topo.no_Capacitors


@pytest.mark.parametrize('name', _CIRCUITS.keys())
def test_Dcut_column_count(name):
    """D_cut must have no_QPS + no_Inductors columns (one-island count)."""
    topo = Topology(_CIRCUITS[name])
    assert topo.D_cut.shape[1] == topo.no_QPS + topo.no_Inductors


@pytest.mark.parametrize('name', _CIRCUITS.keys())
def test_compact_flux_in_ker_Dloop(name):
    """Compact flux columns of Kloop must lie in ker(D_loop)."""
    topo = Topology(_CIRCUITS[name])
    nCF = topo.no_reduced_compact_flux
    if nCF == 0:
        return
    no_two_island = topo.no_JJ + topo.no_Capacitors
    Kloop_compact_two_island = topo.K[:no_two_island, :nCF]
    assert np.allclose(topo.D_loop @ Kloop_compact_two_island, 0), \
        f"{name}: D_loop @ compact_flux != 0"


@pytest.mark.parametrize('name', _CIRCUITS.keys())
def test_compact_charge_in_ker_Dcut(name):
    """Compact charge columns of Kcut must lie in ker(D_cut)."""
    topo = Topology(_CIRCUITS[name])
    nCC = topo.no_reduced_compact_charge
    if nCC == 0:
        return
    ne = topo.no_elements
    no_two_island = topo.no_JJ + topo.no_Capacitors
    n_flux_vars = topo.Fcut.shape[0]
    # Kcut compact charge: rows for one-island elements, columns [n_flux_vars:n_flux_vars+nCC]
    Kcut_compact_one_island = topo.K[ne + no_two_island:, n_flux_vars:n_flux_vars + nCC]
    assert np.allclose(topo.D_cut @ Kcut_compact_one_island, 0), \
        f"{name}: D_cut @ compact_charge != 0"
