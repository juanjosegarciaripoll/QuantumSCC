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
  Compact flux   : transmon (JJ+C), fluxonium (JJ+C+L), N JJ parallel/series
  Compact charge : dual-transmon (QPS+L), dual-fluxonium (QPS+L+C), N QPS parallel/series
  Mixed JJ+QPS   : same nodes, chain, multi-node with shared nodes
  Rings          : 3-node rings with JJ+QPS combinations
  Dualmon Fig. 1 : bare dualmon, gate dualmon, full dualmon
"""

import sys
import os

current_dir  = os.path.dirname(os.path.abspath(__file__))
package_dir  = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC.core.elements import Capacitor, Inductor, Junction, PhaseSlip


def _J():
    return Junction(value=1, unit='GHz')

def _P():
    return PhaseSlip(value=1, unit='GHz')

def _C():
    return Capacitor(value=1, unit='GHz')

def _L():
    return Inductor(value=1, unit='GHz')


CIRCUIT_REGISTRY = [
    ("LC",                 lambda: [(0, 1, _L()), (0, 1, _C())]),
    ("coupled_LC",         lambda: [(0, 1, _L()), (0, 1, _C()), (0, 2, _L()), (0, 2, _C())]),
    ("transmon",           lambda: [(0, 1, _J()), (0, 1, _C())]),
    ("fluxonium",          lambda: [(0, 1, _J()), (0, 1, _C()), (0, 1, _L())]),
    ("2JJ_parallel",       lambda: [(0, 1, _J()), (0, 1, _J()), (0, 1, _C()), (0, 1, _C())]),
    ("2JJ_series",         lambda: [(0, 1, _J()), (0, 1, _C()), (1, 2, _J()), (1, 2, _C()), (0, 2, _C())]),
    ("dual_transmon",      lambda: [(0, 1, _P()), (0, 1, _L())]),
    ("dual_fluxonium",     lambda: [(0, 1, _P()), (0, 1, _L()), (0, 1, _C())]),
    ("2QPS_parallel",      lambda: [(0, 1, _P()), (0, 1, _P()), (0, 1, _L()), (0, 1, _L())]),
    ("2QPS_series",        lambda: [(0, 1, _P()), (0, 1, _L()), (1, 2, _P()), (1, 2, _L()), (0, 2, _L())]),
    ("JJ_QPS_same_nodes",  lambda: [(0, 1, _J()), (0, 1, _C()), (0, 1, _P()), (0, 1, _L())]),
    ("JJ_QPS_chain",       lambda: [(0, 1, _J()), (0, 1, _C()), (1, 2, _P()), (1, 2, _L())]),
    ("2JJ_QPS_shared_node",lambda: [(0, 1, _J()), (0, 1, _C()), (1, 2, _J()), (1, 2, _C()), (0, 1, _P()), (0, 1, _L())]),
    ("JJ_QPS_JJ_chain",    lambda: [(0, 1, _J()), (0, 1, _C()), (1, 2, _P()), (1, 2, _L()), (2, 3, _J()), (2, 3, _C())]),
    ("JJ_JJ_QPS_ring",     lambda: [(0, 1, _J()), (0, 1, _C()), (1, 2, _J()), (1, 2, _C()), (2, 0, _P()), (2, 0, _L())]),
    ("QPS_QPS_JJ_ring",    lambda: [(0, 1, _P()), (0, 1, _L()), (1, 2, _P()), (1, 2, _L()), (2, 0, _J()), (2, 0, _C())]),
    # Dualmon Fig. 1 — bare elements without companions
    ("dualmon_bare",       lambda: [(0, 1, _J()), (0, 1, _P())]),
    # dualmon_full: JJ + C on node (1,0), L series (1,2), QPS + parallel L on (2,0), gate Cx on (1,0)
    ("dualmon_full",       lambda: [(1, 0, _J()), (1, 0, _C()), (1, 2, _L()),
                                    (2, 0, _P()), (2, 0, _L()), (1, 0, _C())]),
]
