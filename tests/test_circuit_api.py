"""
Tests for the Circuit API surface (Tier C).

Covers the parts of circuit.py / the debug print paths that the
architecture-focused test_circuit.py does not exercise:
  * conjugate_pairs() labels, types and ordering;
  * the thin wrapper methods delegating to topo / geom / quant;
  * the debug=True construction path (smoke, via capsys);
  * the numerical printout methods (Hamiltonian_expression,
    diagonal_harmonic_Hamiltonian_expression).
"""

import numpy as np
import pytest

from QuantumSCC import Capacitor, Circuit, Inductor, Junction, PhaseSlip


def transmon():
    return Circuit([(0, 1, Junction(10, "GHz")), (0, 1, Capacitor(0.2, "GHz"))])


def dual_transmon():
    return Circuit([(0, 1, PhaseSlip(5, "GHz")), (0, 1, Inductor(1, "GHz"))])


def lc():
    return Circuit([(0, 1, Inductor(2, "GHz")), (0, 1, Capacitor(0.5, "GHz"))])


def jj_qps_chain():
    return Circuit([(0, 1, Junction(1, "GHz")), (0, 1, Capacitor(1, "GHz")),
                    (1, 2, PhaseSlip(1, "GHz")), (1, 2, Inductor(1, "GHz"))])


# ── conjugate_pairs ─────────────────────────────────────────────────────────

class TestConjugatePairs:
    def test_transmon_single_jj_compact(self):
        assert transmon().conjugate_pairs() == [("phi_c1", "n_c1", "JJ_compact")]

    def test_dual_transmon_single_qps_compact(self):
        assert dual_transmon().conjugate_pairs() == [("psi_c1", "q_c1", "QPS_compact")]

    def test_lc_single_extended(self):
        assert lc().conjugate_pairs() == [("phi_e1", "n_e1", "extended")]

    def test_mixed_chain_ordering(self):
        """Ordering is [compact_flux | compact_charge | extended]."""
        pairs = jj_qps_chain().conjugate_pairs()
        assert pairs == [
            ("phi_c1", "n_c1", "JJ_compact"),
            ("psi_c1", "q_c1", "QPS_compact"),
        ]

    def test_length_equals_number_of_modes(self):
        for factory in (transmon, dual_transmon, lc, jj_qps_chain):
            c = factory()
            assert len(c.conjugate_pairs()) == c.no_independent_variables // 2

    def test_pair_type_values(self):
        types = {t for _, _, t in jj_qps_chain().conjugate_pairs()}
        assert types <= {"JJ_compact", "QPS_compact", "extended"}


# ── wrapper delegation ──────────────────────────────────────────────────────

class TestWrapperDelegation:
    def setup_method(self):
        self.c = jj_qps_chain()

    def test_kirchhoff_wrapper(self):
        result = self.c.Kirchhoff()
        assert len(result) == 6
        np.testing.assert_allclose(result[0], self.c.Fcut)
        np.testing.assert_allclose(result[1], self.c.Floop)

    def test_omega_function_wrapper(self):
        result = self.c.omega_function()
        np.testing.assert_allclose(result[0], self.c.omega_2B)
        np.testing.assert_allclose(result[1], self.c.omega_symplectic)

    def test_classical_hamiltonian_wrapper(self):
        H, vJ, vQ = self.c.classical_hamiltonian_function()
        np.testing.assert_allclose(H, self.c.quadratic_hamiltonian)
        np.testing.assert_allclose(vJ, self.c.vector_JJ)
        np.testing.assert_allclose(vQ, self.c.vector_QPS)

    def test_extended_quantization_wrapper(self):
        H, T, G = self.c.extended_hamiltonian_quantization()
        np.testing.assert_allclose(H, self.c.extended_quantum_hamiltonian)

    def test_total_quantization_wrapper(self):
        result = self.c.total_hamiltonian_quantization()
        assert len(result) == 8
        np.testing.assert_allclose(result[0], self.c.FS_quadratic_hamiltonian_phiq)


# ── debug construction ──────────────────────────────────────────────────────

class TestDebugConstruction:
    @pytest.mark.parametrize("factory", [transmon, dual_transmon, lc, jj_qps_chain],
                             ids=["transmon", "dual_transmon", "lc", "jj_qps_chain"])
    def test_debug_true_prints_and_constructs(self, factory, capsys):
        # Rebuild with debug=True by re-running the factory's elements.
        base = factory()
        c = Circuit(base.elements, debug=True)
        out = capsys.readouterr().out
        # Debug banners from every pipeline stage should appear.
        assert "INITIALIZING CIRCUIT ANALYSIS" in out
        assert "TOPOLOGY" in out
        assert "SYMPLECTIC GEOMETRY" in out
        assert "END OF DEBUGGING" in out
        # Construction still yields a consistent circuit.
        assert c.no_independent_variables == base.no_independent_variables


# ── numerical printouts ─────────────────────────────────────────────────────

class TestNumericalPrintout:
    def test_hamiltonian_expression_nonverbose(self, capsys):
        transmon().Hamiltonian_expression(verbose=False)
        out = capsys.readouterr().out
        assert "Numerical Hamiltonian" in out
        assert "H/" in out

    def test_hamiltonian_expression_verbose(self, capsys):
        transmon().Hamiltonian_expression(verbose=True)
        out = capsys.readouterr().out
        assert "coupling vectors" in out
        assert "Operator subscripts explanation" in out

    def test_hamiltonian_expression_qps_prints_charge_vectors(self, capsys):
        dual_transmon().Hamiltonian_expression(verbose=True)
        out = capsys.readouterr().out
        assert "QPS coupling vectors" in out

    def test_diagonal_harmonic_expression(self, capsys):
        lc().diagonal_harmonic_Hamiltonian_expression()
        out = capsys.readouterr().out
        assert "Diagonalized quantum Hamiltonian" in out

    def test_symbolic_wrapper_returns_none_but_prints(self, capsys):
        # Circuit.symbolic_hamiltonian_expression is a thin printing wrapper.
        assert transmon().symbolic_hamiltonian_expression(verbose=False) is None
        assert "Symbolic Hamiltonian" in capsys.readouterr().out
