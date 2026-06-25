"""
test_verification.py — SVD hidden gauge detection verification (Steps 3b & 3c)

Tests the LC_QPS_coupled circuit (LC oscillator on 0-1 coupled to QPS+L on 1-2)
which exercises the SVD-based hidden gauge detection in linalg.py.

Background
----------
Before the SVD fix, omega_symplectic_transformation failed with
ValueError("linear dependencies between rows of Omega") for this circuit
because the gauge direction was the non-coordinate-aligned combination
(φ_CF − φ_EF)/√2, invisible to the zero-row scan.

The SVD of Ω_FC exposes ker(Ω_FC^T) ⊕ ker(Ω_FC), rotates to align gauges
with coordinates, then deletes them.  After reduction the circuit has:
  • 1 extended oscillator mode (φ_e, n_e)
  • 1 compact QPS mode (ψ_c, q_c) — flux is periodic, charge is integer

Analytical frequency
--------------------
The quadratic Hamiltonian in the reduced (φ, q) basis is diagonal with entries
    [2·E_C, 2·E_L1, 2·E_L2, 0]
The zero corresponds to q_c (compact QPS charge has no quadratic energy).
Hence  max(eigvalsh(FS_quadratic_hamiltonian_phiq)) = 2 · max(E_C, E_L1, E_L2).
"""

import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC import Circuit, Capacitor, Inductor, Junction, PhaseSlip


# ── helpers ───────────────────────────────────────────────────────────────────

def lc_qps_coupled(EC, EL1, EL2, EP=5.0):
    """LC (E_C, E_L1) on nodes 0-1  +  QPS+L (E_P, E_L2) on nodes 1-2."""
    return Circuit([
        (0, 1, Inductor(EL1, 'GHz')),
        (0, 1, Capacitor(EC,  'GHz')),
        (1, 2, PhaseSlip(EP,  'GHz')),
        (1, 2, Inductor(EL2, 'GHz')),
    ])


# ── Step 3b: structural invariants ────────────────────────────────────────────

class TestLCQPSCoupledInvariants:
    """Dedicated structural tests for the LC_QPS_coupled topology.

    This is the canonical example circuit that requires SVD hidden gauge
    detection to construct correctly.
    """

    def setup_method(self):
        self.circ = lc_qps_coupled(EC=1.0, EL1=1.0, EL2=1.0)

    # -- topology ---------------------------------------------------------------

    def test_element_counts(self):
        c = self.circ
        assert c.no_JJ == 0
        assert c.no_QPS == 1
        assert c.no_Capacitors == 1
        assert c.no_Inductors == 2

    def test_no_compact_flux(self):
        """SVD rotation mixes compact+extended flux → compact flux is destroyed."""
        assert self.circ.no_final_compact_flux == 0

    def test_one_compact_charge(self):
        """QPS branch creates exactly one compact charge mode."""
        assert self.circ.no_final_compact_charge == 1

    def test_reduced_compact_flux_zero(self):
        assert self.circ.no_reduced_compact_flux == 0

    def test_reduced_compact_charge_one(self):
        assert self.circ.no_reduced_compact_charge == 1

    def test_no_cap_parallel_qps(self):
        """Cap is on a different branch from QPS — no Cap||QPS conflict."""
        # The circuit builds successfully (no ValueError), confirming
        # Cap on (0,1) and QPS on (1,2) are on different node pairs.
        assert self.circ.no_reduced_compact_charge == 1

    # -- coupling vectors -------------------------------------------------------

    def test_vector_qps_nonzero(self):
        """QPS coupling vector must be non-zero (the mode couples to H)."""
        vq = self.circ.vector_QPS
        assert not np.allclose(vq, 0), "vector_QPS should not be all-zero"

    def test_vector_jj_empty(self):
        """No JJ → vector_JJ is empty."""
        vj = self.circ.vector_JJ
        assert vj.size == 0 or np.allclose(vj, 0)

    # -- Hamiltonian structure --------------------------------------------------

    def test_h_quad_symmetric(self):
        H = self.circ.quadratic_hamiltonian.real
        np.testing.assert_allclose(H, H.T, atol=1e-12,
                                   err_msg="quadratic_hamiltonian not symmetric")

    def test_h_quad_psd(self):
        H = self.circ.quadratic_hamiltonian.real
        eigs = np.linalg.eigvalsh(H)
        assert np.all(eigs >= -1e-12), f"H_quad has negative eigenvalue: {eigs.min():.4e}"

    def test_fs_h_phiq_symmetric(self):
        H = self.circ.FS_quadratic_hamiltonian_phiq.real
        np.testing.assert_allclose(H, H.T, atol=1e-12,
                                   err_msg="FS_quadratic_hamiltonian_phiq not symmetric")

    def test_fs_h_phiq_psd(self):
        H = self.circ.FS_quadratic_hamiltonian_phiq.real
        eigs = np.linalg.eigvalsh(H)
        assert np.all(eigs >= -1e-12), f"FS H_phiq has negative eigenvalue: {eigs.min():.4e}"

    def test_h_quad_one_zero_eigenvalue(self):
        """Compact QPS charge (q_c) contributes zero to quadratic H."""
        H = self.circ.quadratic_hamiltonian.real
        eigs = sorted(np.linalg.eigvalsh(H))
        n_zeros = sum(1 for e in eigs if abs(e) < 1e-10)
        assert n_zeros == 1, f"Expected exactly 1 zero eigenvalue, got {n_zeros}"

    # -- FK = 0 (Kirchhoff) ----------------------------------------------------

    def test_fk_zero(self):
        topo = self.circ.topo
        assert np.allclose(topo.F @ topo.K, 0), "F @ K != 0"

    # -- Darboux condition: omega_symplectic is canonical J ─────────────────────

    def test_omega_symplectic_canonical(self):
        """omega_symplectic after gauge reduction should be block [[0,I],[-I,0]]."""
        Omega = self.circ.omega_symplectic.real
        n = Omega.shape[0]
        assert n % 2 == 0
        k = n // 2
        J = np.block([[np.zeros((k, k)), np.eye(k)],
                      [-np.eye(k), np.zeros((k, k))]])
        np.testing.assert_allclose(Omega, J, atol=1e-10,
                                   err_msg="omega_symplectic is not canonical J")


# ── Step 3c: parametric frequency — ω = 2·max(E_C, E_L1, E_L2) ──────────────

LC_QPS_PARAMS = [
    # (EC, EL1, EL2)
    (1.0, 1.0, 1.0),   # balanced: max=EC=EL1=EL2=1
    (1.0, 2.0, 1.0),   # EL1 dominates
    (1.0, 1.0, 2.0),   # EL2 dominates
    (2.0, 1.0, 1.0),   # EC dominates
    (1.0, 3.0, 2.0),   # EL1 > EL2 > EC
    (0.5, 2.0, 3.0),   # EL2 > EL1 > EC
    (3.0, 1.0, 1.0),   # EC large
    (1.0, 4.0, 2.0),   # EL1 large
    (2.0, 2.0, 2.0),   # all equal, large
    (0.5, 0.5, 2.0),   # EL2 unique max
    (4.0, 1.0, 1.0),   # EC very large
    (1.0, 1.0, 4.0),   # EL2 very large
    (2.0, 3.0, 1.0),   # EL1 > EC > EL2
    (1.0, 2.0, 3.0),   # EL2 > EL1 = EC
]


@pytest.mark.parametrize("EC,EL1,EL2", LC_QPS_PARAMS,
    ids=[f"EC{a}_EL1{b}_EL2{c}" for a, b, c in LC_QPS_PARAMS])
def test_lc_qps_coupled_frequency(EC, EL1, EL2):
    """LC_QPS_coupled frequency: max(eigvalsh(FS_H)) = 2·max(E_C, E_L1, E_L2).

    The quadratic Hamiltonian diagonal in the reduced (φ,q) basis is
    [2·E_C, 2·E_L1, 2·E_L2, 0], so the maximum eigenvalue equals
    2·max(E_C, E_L1, E_L2).  This verifies correct gauge reduction via SVD.
    """
    circ = lc_qps_coupled(EC, EL1, EL2)

    H = circ.FS_quadratic_hamiltonian_phiq.real
    omega_num = max(np.linalg.eigvalsh(H))
    omega_th  = 2.0 * max(EC, EL1, EL2)

    np.testing.assert_allclose(
        omega_num, omega_th, rtol=1e-6,
        err_msg=f"EC={EC}, EL1={EL1}, EL2={EL2}: ω={omega_num:.6f} ≠ {omega_th:.6f}"
    )


@pytest.mark.parametrize("EC,EL1,EL2", LC_QPS_PARAMS,
    ids=[f"EC{a}_EL1{b}_EL2{c}" for a, b, c in LC_QPS_PARAMS])
def test_lc_qps_coupled_structural_invariants(EC, EL1, EL2):
    """LC_QPS_coupled: structural invariants hold across all parameter values."""
    circ = lc_qps_coupled(EC, EL1, EL2)
    topo = circ.topo

    # Kirchhoff
    assert np.allclose(topo.F @ topo.K, 0), "F @ K != 0"

    # Compact mode structure
    assert circ.no_final_compact_flux == 0, "nCF should be 0"
    assert circ.no_final_compact_charge == 1, "nCC should be 1"

    # Coupling vector non-zero
    assert not np.allclose(circ.vector_QPS, 0), "vector_QPS all zero"

    # Quadratic H symmetric + PSD
    H = circ.quadratic_hamiltonian.real
    np.testing.assert_allclose(H, H.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(H) >= -1e-12)
