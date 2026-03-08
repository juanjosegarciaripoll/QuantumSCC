"""
conftest.py — shared definitions for the QuantumSCC test suite.

CIRCUIT_REGISTRY
----------------
Reference catalogue of all supported topologies (name, factory pairs).
factory() returns a fresh list of (node_a, node_b, element) tuples.

This list documents which circuit families the library supports.
It is used by test_circuit.py for concrete-circuit tests.
Unit test files (test_topology, test_geometry, test_quantization) each
define their own single representative circuit — no parametrization.

Topologies covered
------------------
  Linear         : LC, coupled oscillators
  Compact flux   : transmon (JJ), fluxonium (JJ+L), N JJ parallel/series
  Compact charge : dual-transmon (QPS), dual-fluxonium (QPS+C), N QPS parallel/series
  Mixed JJ+QPS   : same nodes (JJ‖QPS), chain (JJ-QPS), multi-node with shared nodes
  Rings          : 3-node rings with JJ+QPS combinations
"""

import sys
import os

current_dir  = os.path.dirname(os.path.abspath(__file__))
package_dir  = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC.core.elements import Capacitor, Inductor, Junction, PhaseSlip


def _J():
    return Junction(value=1, unit='GHz', cap=Capacitor(value=1, unit='GHz'))

def _P():
    return PhaseSlip(value=1, unit='GHz', L_value=1, L_unit='GHz')

def _C():
    return Capacitor(value=1, unit='GHz')

def _L():
    return Inductor(value=1, unit='GHz')


CIRCUIT_REGISTRY = [
    ("LC",                 lambda: [(0, 1, _L()), (0, 1, _C())]),
    ("coupled_LC",         lambda: [(0, 1, _L()), (0, 1, _C()), (0, 2, _L()), (0, 2, _C())]),
    ("transmon",           lambda: [(0, 1, _J())]),
    ("fluxonium",          lambda: [(0, 1, _J()), (0, 1, _L())]),
    ("2JJ_parallel",       lambda: [(0, 1, _J()), (0, 1, _J())]),
    ("2JJ_series",         lambda: [(0, 1, _J()), (1, 2, _J()), (0, 2, _C())]),
    ("dual_transmon",      lambda: [(0, 1, _P())]),
    ("dual_fluxonium",     lambda: [(0, 1, _P()), (0, 1, _C())]),
    ("2QPS_parallel",      lambda: [(0, 1, _P()), (0, 1, _P())]),
    ("2QPS_series",        lambda: [(0, 1, _P()), (1, 2, _P()), (0, 2, _L())]),
    ("JJ_QPS_same_nodes",  lambda: [(0, 1, _J()), (0, 1, _P())]),
    ("JJ_QPS_chain",       lambda: [(0, 1, _J()), (1, 2, _P())]),
    ("2JJ_QPS_shared_node",lambda: [(0, 1, _J()), (1, 2, _J()), (0, 1, _P())]),
    ("JJ_QPS_JJ_chain",    lambda: [(0, 1, _J()), (1, 2, _P()), (2, 3, _J())]),
    ("JJ_JJ_QPS_ring",     lambda: [(0, 1, _J()), (1, 2, _J()), (2, 0, _P())]),
    ("QPS_QPS_JJ_ring",    lambda: [(0, 1, _P()), (1, 2, _P()), (2, 0, _J())]),
]
