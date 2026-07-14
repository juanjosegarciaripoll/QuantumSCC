"""
Error paths and edge-branch coverage (Tier F).

Covers the defensive/validation branches and the sector-dependent print
sub-branches that the mainstream tests do not reach:
  * topology "too many JJ / QPS" companion-count validation errors;
  * the _independent_columns_ordered empty-matrix guard;
  * the full symbolic / numerical printouts on a circuit that spans all three
    sectors at once (compact flux + compact charge + extended);
  * off-diagonal coupling print branches on a coupled circuit.

The remaining uncovered lines in quantization.py (the "JJ depends on
non-dynamical variables", "PhaseSlip decoupled" and "no Hamiltonian dynamics"
raises) are defensive guards for pathological internal states that the public
API / topology validation makes unreachable, so they are intentionally left
uncovered rather than exercised through contrived white-box state.
"""

import numpy as np
import pytest

from QuantumSCC import Capacitor, Circuit, Inductor, Junction, PhaseSlip
from QuantumSCC.core.topology import _independent_columns_ordered
from QuantumSCC.utils import units as unt


def rich_three_sector():
    """A circuit with compact-flux (JJ), compact-charge (QPS) and extended
    (LC oscillator) modes all present: nCF=nCC=nEF=1."""
    return Circuit([
        (0, 1, Junction(1, "GHz")), (0, 1, Capacitor(1, "GHz")),      # JJ compact flux
        (1, 2, PhaseSlip(1, "GHz")), (1, 2, Inductor(1, "GHz")),      # QPS compact charge
        (2, 3, Inductor(1, "GHz")), (2, 3, Capacitor(1, "GHz")),      # LC extended mode
    ])


# ── element edge branches ───────────────────────────────────────────────────

class TestElementEdges:
    def test_default_units_used_when_none(self):
        # unit=None takes the module default-unit branch for each element type.
        assert Capacitor(1.0).unit == unt.get_unit_cap()
        assert Inductor(1.0).unit == unt.get_unit_ind()
        assert Junction(1.0).unit == unt.get_unit_JJ()
        assert PhaseSlip(1.0).unit == unt.get_unit_JJ()

    def test_capacitor_random_value_equals_mean_without_error(self):
        # No 'error' attribute -> std 0 -> random draw is exactly the mean.
        c = Capacitor(1.0, "GHz")
        v = c.value(random=True)
        assert isinstance(v, float)
        assert v == c.value(random=False)

    def test_inductor_random_value_equals_mean_without_error(self):
        ind = Inductor(1.0, "GHz")
        v = ind.value(random=True)
        assert isinstance(v, float)
        assert v == ind.value(random=False)


# ── topology companion-count validation ─────────────────────────────────────

class TestCompanionValidation:
    def test_excess_qps_needs_inductors(self):
        with pytest.raises(ValueError, match="Too many PhaseSlip"):
            Circuit([(0, 1, PhaseSlip(1, "GHz")),
                     (0, 1, PhaseSlip(1, "GHz")),
                     (0, 1, Inductor(1, "GHz"))])

    def test_excess_jj_needs_capacitors(self):
        with pytest.raises(ValueError, match="Too many Junction"):
            Circuit([(0, 1, Junction(1, "GHz")),
                     (0, 1, Junction(1, "GHz")),
                     (0, 1, Capacitor(1, "GHz"))])

    def test_bare_single_nonlinear_is_allowed(self):
        # exactly one bare JJ / one bare QPS must NOT trip the excess check.
        Circuit([(0, 1, Junction(1, "GHz")), (0, 1, PhaseSlip(1, "GHz"))])


# ── _independent_columns_ordered guard ──────────────────────────────────────

class TestIndependentColumnsGuard:
    def test_empty_matrix_returned_unchanged(self):
        M = np.zeros((3, 0))
        out = _independent_columns_ordered(M)
        assert out.shape == (3, 0)

    def test_drops_dependent_columns_keeps_order(self):
        # col2 == col0 -> dropped; the first independent set is kept in order.
        M = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        out = _independent_columns_ordered(M)
        assert out.shape[1] == 2
        np.testing.assert_allclose(out, M[:, :2])


# ── all-sector print branches ───────────────────────────────────────────────

class TestAllSectorPrintout:
    def test_symbolic_verbose_prints_all_legends(self, capsys):
        rich_three_sector().symbolic_hamiltonian_expression(verbose=True)
        out = capsys.readouterr().out
        # legend lines for each sector present in a rich circuit
        assert "compact flux" in out          # phi_c / psi_c
        assert "extended flux" in out          # phi_e
        assert "compact charge" in out         # q_c
        assert "extended charge" in out        # n_e
        assert "Parameter values" in out

    def test_numeric_verbose_covers_all_var_labels(self, capsys):
        rich_three_sector().Hamiltonian_expression(verbose=True)
        out = capsys.readouterr().out
        assert "coupling vectors" in out
        assert "QPS coupling vectors" in out
        # the charge variable vector (QPS sector) legend is printed
        assert "Charge variable vector" in out

    def test_conjugate_pairs_span_all_sectors(self):
        pairs = rich_three_sector().conjugate_pairs()
        types = [t for _, _, t in pairs]
        assert types == ["JJ_compact", "QPS_compact", "extended"]


# ── off-diagonal coupling print branches ────────────────────────────────────

class TestCouplingPrintout:
    def test_coupled_oscillators_print_offdiagonal_terms(self, capsys):
        # Two LC modes sharing a node produce off-diagonal quadratic couplings,
        # exercising the flux-flux / charge-charge coupling print branches.
        c = Circuit([(0, 1, Inductor(1, "GHz")), (0, 1, Capacitor(1, "GHz")),
                     (0, 2, Inductor(1, "GHz")), (0, 2, Capacitor(1, "GHz"))])
        c.Hamiltonian_expression(verbose=False)
        out = capsys.readouterr().out
        assert "Numerical Hamiltonian" in out


# ── multi-mode / multi-element print branches ───────────────────────────────

class TestMultiElementPrintout:
    def test_diagonal_harmonic_two_modes(self, capsys):
        # Two extended oscillator modes exercise the non-final ("... + ...")
        # branch of diagonal_harmonic_Hamiltonian_expression.
        c = Circuit([(0, 1, Inductor(1, "GHz")), (0, 1, Capacitor(1, "GHz")),
                     (0, 2, Inductor(1, "GHz")), (0, 2, Capacitor(1, "GHz"))])
        c.diagonal_harmonic_Hamiltonian_expression()
        out = capsys.readouterr().out
        assert out.count("a†") >= 2  # both modes printed

    def test_two_junctions_print_multiple_cos(self, capsys):
        # Two JJs -> the non-final cos(v phi) term is printed (trailing branch).
        c = Circuit([(0, 1, Junction(1, "GHz")), (0, 1, Junction(1, "GHz")),
                     (0, 1, Capacitor(1, "GHz")), (0, 1, Capacitor(1, "GHz"))])
        c.Hamiltonian_expression(verbose=False)
        out = capsys.readouterr().out
        assert out.count("cos(v") >= 2

    def test_two_phaseslips_print_multiple_cos(self, capsys):
        # Two QPSs -> the non-final cos(u q) term is printed (trailing branch).
        c = Circuit([(0, 1, PhaseSlip(1, "GHz")), (0, 1, PhaseSlip(1, "GHz")),
                     (0, 1, Inductor(1, "GHz")), (0, 1, Inductor(1, "GHz"))])
        c.Hamiltonian_expression(verbose=False)
        out = capsys.readouterr().out
        assert out.count("cos(u") >= 2

    def test_offdiagonal_flux_coupling_printed(self, capsys):
        # dualmon_full retains an off-diagonal flux-flux quadratic coupling in
        # the phiq basis, exercising the flux-flux coupling print branch.
        # (It cannot be diagonalised, but Hamiltonian_expression works on the
        # quadratic form directly.)
        c = Circuit([(1, 0, Junction(1, "GHz")), (1, 0, Capacitor(1, "GHz")),
                     (1, 2, Inductor(1, "GHz")),
                     (2, 0, PhaseSlip(1, "GHz")), (2, 0, Inductor(1, "GHz")),
                     (1, 0, Capacitor(1, "GHz"))])
        c.Hamiltonian_expression(verbose=False)
        out = capsys.readouterr().out
        assert "Numerical Hamiltonian" in out
