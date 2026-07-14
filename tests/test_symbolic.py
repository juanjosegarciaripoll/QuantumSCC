"""
Tests for QuantumSCC.utils.symbolic and the symbolic Hamiltonian printout.

Covers:
  * _to_sym coefficient cleaning (rounding, zero threshold, radical recovery);
  * build_symbolic_hamiltonian structural outputs (shape, symbol counts,
    symmetry) across the whole circuit registry;
  * the symbolic <-> numerical consistency relation
        eigvals(H_sym.subs(vals)) == eigvals(quadratic_hamiltonian)
    Both pipelines use the Adrián convention: M matrix from H = ½ ξᵀ M ξ;
  * the Circuit.symbolic_hamiltonian_expression printout (terminal path).
"""

import numpy as np
import pytest
import sympy as sp
from conftest import CIRCUIT_REGISTRY

from QuantumSCC import Capacitor, Circuit, Inductor, Junction, PhaseSlip
from QuantumSCC.core.geometry import Geometry
from QuantumSCC.core.quantization import Quantization
from QuantumSCC.core.topology import Topology
from QuantumSCC.utils.symbolic import _to_sym, build_symbolic_hamiltonian


def _build(factory):
    """Return (topo, geom, quant) for a registry factory."""
    topo = Topology(factory())
    geom = Geometry(topo)
    quant = Quantization(topo, geom)
    return topo, geom, quant


# ── _to_sym ────────────────────────────────────────────────────────────────

class TestToSym:
    def test_exact_zero(self):
        assert _to_sym(0.0) == sp.Integer(0)

    def test_below_tolerance_is_zero(self):
        assert _to_sym(1e-12) == sp.Integer(0)

    def test_integer_passthrough(self):
        assert _to_sym(3.0) == sp.Integer(3)

    def test_negative_integer(self):
        assert _to_sym(-2.0) == sp.Integer(-2)

    def test_near_integer_rounds(self):
        assert _to_sym(2.9999999999) == sp.Integer(3)

    def test_returns_sympy_expression(self):
        assert isinstance(_to_sym(1.5), sp.Basic)

    def test_recovers_sqrt2(self):
        # nsimplify with the sqrt(2) basis should recover the radical exactly.
        assert sp.simplify(_to_sym(float(np.sqrt(2))) - sp.sqrt(2)) == 0

    def test_custom_tolerance(self):
        # 0.4 is not < tol=0.1, so it is not collapsed to 0.
        assert _to_sym(0.4, tol=0.1) != sp.Integer(0)
        # 0.05 is < tol=0.1 -> 0.
        assert _to_sym(0.05, tol=0.1) == sp.Integer(0)


# ── build_symbolic_hamiltonian: structure ───────────────────────────────────

class TestBuildSymbolicStructure:
    def test_shape_equals_no_indep(self):
        topo, geom, _ = _build(lambda: [(0, 1, Junction(1, "GHz")),
                                        (0, 1, Capacitor(1, "GHz"))])
        H_sym, *_ = build_symbolic_hamiltonian(topo, geom)
        assert H_sym.shape == (geom.no_independent_variables,
                               geom.no_independent_variables)

    def test_symbol_counts(self):
        # 1 JJ + 1 Cap + 1 QPS + 1 Ind
        topo, geom, _ = _build(lambda: [(0, 1, Junction(1, "GHz")),
                                        (0, 1, Capacitor(1, "GHz")),
                                        (1, 2, PhaseSlip(1, "GHz")),
                                        (1, 2, Inductor(1, "GHz"))])
        _, sym_vals, J_syms, P_syms = build_symbolic_hamiltonian(topo, geom)
        assert len(J_syms) == topo.no_JJ == 1
        assert len(P_syms) == topo.no_QPS == 1
        # one energy symbol per Capacitor and per Inductor
        assert len(sym_vals) == topo.no_Capacitors + topo.no_Inductors

    def test_symmetric(self):
        topo, geom, _ = _build(lambda: [(0, 1, Junction(1, "GHz")),
                                        (0, 1, Capacitor(1.3, "GHz")),
                                        (1, 2, PhaseSlip(1, "GHz")),
                                        (1, 2, Inductor(0.7, "GHz"))])
        H_sym, *_ = build_symbolic_hamiltonian(topo, geom)
        assert sp.simplify(H_sym - H_sym.T) == sp.zeros(*H_sym.shape)

    def test_symbol_values_are_twice_energy(self):
        # sym_vals stores E = 2 * element.energy() (Adrián convention).
        topo, geom, _ = _build(lambda: [(0, 1, Inductor(2, "GHz")),
                                        (0, 1, Capacitor(0.5, "GHz"))])
        _, sym_vals, _, _ = build_symbolic_hamiltonian(topo, geom)
        for elem in (e[2] for e in topo.elements):
            if isinstance(elem, (Capacitor, Inductor)):
                assert any(abs(v - 2 * elem.energy()) < 1e-9
                           for v in sym_vals.values())

    def test_jj_and_qps_values_match_elements(self):
        topo, geom, _ = _build(lambda: [(0, 1, Junction(3.0, "GHz")),
                                        (0, 1, Capacitor(1, "GHz")),
                                        (1, 2, PhaseSlip(4.0, "GHz")),
                                        (1, 2, Inductor(1, "GHz"))])
        _, _, J_syms, P_syms = build_symbolic_hamiltonian(topo, geom)
        assert abs(J_syms[0][1] - 3.0) < 1e-9
        assert abs(P_syms[0][1] - 4.0) < 1e-9


# ── symbolic <-> numerical consistency ────────────────────────────────────────

@pytest.mark.parametrize("name,factory", CIRCUIT_REGISTRY,
                         ids=[n for n, _ in CIRCUIT_REGISTRY])
def test_symbolic_matches_numeric(name, factory):
    """eigvals(H_sym.subs(vals)) == eigvals(quadratic_hamiltonian).

    Both pipelines now use the same convention: they store the matrix M
    from the PRX 2025 Hamiltonian H = ½ ξᵀ M ξ.  Diagonal entries are
    2·E_C and 2·E_L (Adrián convention).
    """
    topo, geom, quant = _build(factory)
    H_sym, sym_vals, _, _ = build_symbolic_hamiltonian(topo, geom)

    H_num = np.array(H_sym.subs(sym_vals)).astype(float)
    # Substituted symbolic H must be real and symmetric.
    np.testing.assert_allclose(H_num, H_num.T, atol=1e-9)

    eig_sym = np.sort(np.linalg.eigvalsh((H_num + H_num.T) / 2))
    eig_num = np.sort(np.linalg.eigvalsh(quant.quadratic_hamiltonian.real))
    np.testing.assert_allclose(eig_sym, eig_num, atol=1e-6)


@pytest.mark.parametrize("factory", [
    # Series chains have non-dynamical variables -> Schur complement path.
    lambda: [(0, 1, Junction(1, "GHz")), (0, 1, Capacitor(1, "GHz")),
             (1, 2, Junction(1, "GHz")), (1, 2, Capacitor(1, "GHz")),
             (0, 2, Capacitor(1, "GHz"))],
    lambda: [(0, 1, PhaseSlip(1, "GHz")), (0, 1, Inductor(1, "GHz")),
             (1, 2, PhaseSlip(1, "GHz")), (1, 2, Inductor(1, "GHz")),
             (0, 2, Inductor(1, "GHz"))],
])
def test_schur_complement_branch(factory):
    """Circuits with non-dynamical variables exercise the symbolic Schur path."""
    topo, geom, quant = _build(factory)
    H_sym, sym_vals, _, _ = build_symbolic_hamiltonian(topo, geom)
    # H_full was larger than no_indep -> Schur reduction ran and produced a
    # matrix of the reduced size.
    assert H_sym.shape == (geom.no_independent_variables,) * 2
    H_num = np.array(H_sym.subs(sym_vals)).astype(float)
    eig_sym = np.sort(np.linalg.eigvalsh((H_num + H_num.T) / 2))
    eig_num = np.sort(np.linalg.eigvalsh(quant.quadratic_hamiltonian.real))
    np.testing.assert_allclose(eig_sym, eig_num, atol=1e-6)


# ── symbolic_hamiltonian_expression printout ────────────────────────────────

class TestSymbolicPrintout:
    def _transmon(self):
        return Circuit([(0, 1, Junction(10, "GHz")), (0, 1, Capacitor(0.2, "GHz"))])

    def test_quant_returns_sympy_expression(self):
        c = self._transmon()
        expr = c.quant.symbolic_hamiltonian_expression(verbose=False)
        assert isinstance(expr, sp.Basic)
        # E_J and E_C should appear as free symbols.
        names = {str(s) for s in expr.free_symbols}
        assert "E_J" in names
        assert "E_C" in names

    def test_terminal_output_nonverbose(self, capsys):
        c = self._transmon()
        c.symbolic_hamiltonian_expression(verbose=False)
        out = capsys.readouterr().out
        assert "Symbolic Hamiltonian" in out
        assert "H/" in out  # the "H/ℏ = ..." line

    def test_terminal_output_verbose(self, capsys):
        c = self._transmon()
        c.symbolic_hamiltonian_expression(verbose=True)
        out = capsys.readouterr().out
        assert "Canonical variables" in out
        assert "Parameter values" in out
        assert "Numerical Hamiltonian" in out

    def test_qps_circuit_prints_E_P(self, capsys):
        c = Circuit([(0, 1, PhaseSlip(5, "GHz")), (0, 1, Inductor(1, "GHz"))])
        expr = c.quant.symbolic_hamiltonian_expression(verbose=False)
        names = {str(s) for s in expr.free_symbols}
        assert "E_P" in names
        assert "Symbolic Hamiltonian" in capsys.readouterr().out
