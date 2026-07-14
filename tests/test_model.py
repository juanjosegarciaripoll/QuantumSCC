"""
Tests for QuantumSCC.model.CircuitModel and Circuit.model().

Covers:
1. Data extraction — counts, matrices, coupling vectors pinned for known circuits
2. API helpers — sectors, labels, repr, package export
3. Structural invariants — symmetry, no flux-charge cross terms, periodic-operator rule
4. Reconstruction regression — sparse Hamiltonian spectra against known physics
"""

import unittest

import numpy as np

from QuantumSCC import (
    Capacitor,
    Circuit,
    CircuitModel,
    Inductor,
    Junction,
    PhaseSlip,
)


def transmon():
    return Circuit([(0, 1, Junction(10, "GHz")), (0, 1, Capacitor(0.2, "GHz"))])


def lc():
    return Circuit([(0, 1, Inductor(2, "GHz")), (0, 1, Capacitor(0.5, "GHz"))])


def coupled_lc():
    return Circuit([(0, 1, Inductor(1, "GHz")), (0, 1, Capacitor(0.5, "GHz")),
                    (0, 2, Inductor(1, "GHz")), (0, 2, Capacitor(0.5, "GHz"))])


def transmon_resonator():
    return Circuit([(0, 1, Junction(10, "GHz")), (0, 1, Capacitor(0.2, "GHz")),
                    (0, 2, Inductor(1, "GHz")), (0, 2, Capacitor(0.5, "GHz")),
                    (1, 2, Capacitor(0.3, "GHz"))])


def dual_transmon():
    return Circuit([(0, 1, PhaseSlip(5, "GHz")), (0, 1, Inductor(1, "GHz"))])


class TestDataExtraction(unittest.TestCase):
    def test_transmon(self):
        m = transmon().model()
        self.assertEqual((m.n_compact_flux, m.n_compact_charge, m.n_extended), (1, 0, 0))
        np.testing.assert_allclose(m.K_flux, [[0.0]], atol=1e-12)
        np.testing.assert_allclose(m.K_charge, [[0.4]], atol=1e-9)
        np.testing.assert_allclose(m.E_J, [10.0])
        np.testing.assert_allclose(m.v_J, [[1.0]], atol=1e-9)
        self.assertEqual(m.E_P.size, 0)
        self.assertEqual(m.v_P.shape, (0, 1))

    def test_lc_is_extended_oscillator(self):
        m = lc().model()
        self.assertEqual((m.n_compact_flux, m.n_compact_charge, m.n_extended), (0, 0, 1))
        # extended mode is isotropic: K_flux[m,m] == K_charge[m,m] == ω
        np.testing.assert_allclose(m.K_flux, [[2.0]], atol=1e-9)
        np.testing.assert_allclose(m.K_charge, [[2.0]], atol=1e-9)
        self.assertEqual(m.E_J.size, 0)
        self.assertEqual(m.E_P.size, 0)

    def test_coupled_lc_normal_modes(self):
        m = coupled_lc().model()
        self.assertEqual(m.mode_sectors, ("extended", "extended"))
        # diagonalised into decoupled normal modes (no off-diagonal)
        np.testing.assert_allclose(m.K_flux, np.sqrt(2) * np.eye(2), atol=1e-9)
        np.testing.assert_allclose(m.K_charge, np.sqrt(2) * np.eye(2), atol=1e-9)

    def test_transmon_resonator_cross_coupling(self):
        m = transmon_resonator().model()
        self.assertEqual((m.n_compact_flux, m.n_compact_charge, m.n_extended), (1, 0, 1))
        # capacitive coupling survives as a single off-diagonal charge term
        np.testing.assert_allclose(m.K_charge[0, 1], -0.282842712, atol=1e-6)
        np.testing.assert_allclose(m.K_charge[1, 0], -0.282842712, atol=1e-6)
        # JJ couples only to its own compact flux
        np.testing.assert_allclose(np.abs(m.v_J), [[1.0, 0.0]], atol=1e-9)

    def test_dual_transmon_qps(self):
        m = dual_transmon().model()
        self.assertEqual((m.n_compact_flux, m.n_compact_charge, m.n_extended), (0, 1, 0))
        # QPS sector: inductive energy on the (integer) flux ψ_c, none on charge q_c
        np.testing.assert_allclose(m.K_flux, [[2.0]], atol=1e-9)
        np.testing.assert_allclose(m.K_charge, [[0.0]], atol=1e-12)
        np.testing.assert_allclose(m.E_P, [5.0])
        np.testing.assert_allclose(np.abs(m.v_P), [[1.0]], atol=1e-9)
        self.assertEqual(m.E_J.size, 0)


class TestAPI(unittest.TestCase):
    def test_is_circuitmodel_and_exported(self):
        self.assertIsInstance(transmon().model(), CircuitModel)

    def test_n_modes_and_sectors(self):
        m = transmon_resonator().model()
        self.assertEqual(m.n_modes, 2)
        self.assertEqual(m.sector(0), "compact_flux")
        self.assertEqual(m.sector(1), "extended")
        self.assertEqual(m.mode_sectors, ("compact_flux", "extended"))

    def test_mode_labels(self):
        m = transmon_resonator().model()
        self.assertEqual(m.mode_label(0), ("phi_c1", "n_c1"))
        self.assertEqual(m.mode_label(1), ("phi_e1", "n_e1"))
        self.assertEqual(dual_transmon().model().mode_label(0), ("psi_c1", "q_c1"))

    def test_sector_out_of_range(self):
        m = transmon().model()
        with self.assertRaises(IndexError):
            m.sector(5)

    def test_repr(self):
        r = repr(transmon().model())
        self.assertIn("CircuitModel", r)
        self.assertIn("n_modes=1", r)

    def test_from_circuit_equivalent_to_method(self):
        c = transmon()
        np.testing.assert_array_equal(c.model().K_charge, CircuitModel.from_circuit(c).K_charge)

    def test_tol_zero_keeps_raw(self):
        # default sparsifies tiny entries; tol=0 keeps them
        m_sparse = transmon_resonator().model()
        m_raw = transmon_resonator().model(tol=0)
        self.assertTrue(np.all(np.abs(m_sparse.K_flux[m_sparse.K_flux == 0]) == 0))
        # both agree on the physically significant entries
        np.testing.assert_allclose(m_sparse.K_charge, m_raw.K_charge, atol=1e-9)


class TestInvariants(unittest.TestCase):
    CIRCUITS = [transmon, lc, coupled_lc, transmon_resonator, dual_transmon]

    def test_quadratic_symmetric(self):
        for f in self.CIRCUITS:
            m = f().model()
            np.testing.assert_allclose(m.K_flux, m.K_flux.T, atol=1e-12)
            np.testing.assert_allclose(m.K_charge, m.K_charge.T, atol=1e-12)

    def test_periodic_operators_absent_from_quadratic(self):
        # K_flux[m,m]=0 for compact_flux modes; K_charge[m,m]=0 for compact_charge modes
        for f in self.CIRCUITS:
            m = f().model()
            for i in range(m.n_modes):
                if m.sector(i) == "compact_flux":
                    self.assertAlmostEqual(m.K_flux[i, i], 0.0, places=10)
                if m.sector(i) == "compact_charge":
                    self.assertAlmostEqual(m.K_charge[i, i], 0.0, places=10)

    def test_shapes_consistent(self):
        for f in self.CIRCUITS:
            m = f().model()
            N = m.n_modes
            self.assertEqual(m.K_flux.shape, (N, N))
            self.assertEqual(m.K_charge.shape, (N, N))
            self.assertEqual(m.v_J.shape, (m.E_J.size, N))
            self.assertEqual(m.v_P.shape, (m.E_P.size, N))


class TestReconstruction(unittest.TestCase):
    """Regression guard on the documented reconstruction recipe."""

    @staticmethod
    def _reconstruct(model, charge_cut=12, fock_dim=12):
        import scipy.sparse as sp
        from scipy.linalg import expm as dense_expm

        secs = model.mode_sectors
        dims = [fock_dim if s == "extended" else 2 * charge_cut + 1 for s in secs]
        N, dim = model.n_modes, int(np.prod(dims))

        def embed(op, m):
            out = sp.identity(1, format="csc", dtype=complex)
            for k in range(N):
                out = sp.kron(out, op if k == m else sp.identity(dims[k]), format="csc")
            return out

        Phi, Nop = [None] * N, [None] * N
        expF, expN = [None] * N, [None] * N
        for m, s in enumerate(secs):
            d = dims[m]
            if s == "extended":
                a = sp.diags(np.sqrt(np.arange(1, d)), 1, format="csc")
                ad = a.conj().T
                phi, n = (a + ad) / np.sqrt(2), 1j * (ad - a) / np.sqrt(2)
                Phi[m], Nop[m] = phi, n
                expF[m] = lambda c, o=phi: sp.csc_matrix(dense_expm(1j * c * o.toarray()))
                expN[m] = lambda c, o=n: sp.csc_matrix(dense_expm(1j * c * o.toarray()))
            elif s == "compact_flux":
                vals = np.arange(-charge_cut, charge_cut + 1)
                Nop[m] = sp.diags(vals, 0, format="csc", dtype=float)
                shift = sp.diags(np.ones(d - 1), -1, format="csc")
                expF[m] = lambda c, S=shift: (S ** int(round(c)) if c >= 0
                                              else (S.conj().T) ** int(round(-c)))
                expN[m] = lambda c, v=vals: sp.diags(np.exp(1j * c * v), 0, format="csc")
            else:
                vals = np.arange(-charge_cut, charge_cut + 1)
                Phi[m] = sp.diags(vals, 0, format="csc", dtype=float)
                shift = sp.diags(np.ones(d - 1), -1, format="csc")
                expN[m] = lambda c, S=shift: (S ** int(round(c)) if c >= 0
                                              else (S.conj().T) ** int(round(-c)))
                expF[m] = lambda c, v=vals: sp.diags(np.exp(1j * c * v), 0, format="csc")

        H = sp.csc_matrix((dim, dim), dtype=complex)
        for Kmat, ops in ((model.K_flux, Phi), (model.K_charge, Nop)):
            for a in range(N):
                for b in range(N):
                    if abs(Kmat[a, b]) < 1e-12:
                        continue
                    H = H + 0.5 * Kmat[a, b] * (embed(ops[a], a) @ embed(ops[b], b))
        for E, V, expo in ((model.E_J, model.v_J, expF), (model.E_P, model.v_P, expN)):
            for t in range(len(E)):
                P = sp.identity(dim, format="csc", dtype=complex)
                for m in range(N):
                    if abs(V[t, m]) >= 1e-12:
                        P = P @ embed(expo[m](V[t, m]), m)
                H = H - E[t] * 0.5 * (P + P.conj().T)
        return H

    @staticmethod
    def _levels(H, k=4):
        from scipy.sparse.linalg import eigsh
        return np.sort(eigsh(H, k=k, which="SA", return_eigenvectors=False).real)

    def test_lc_spectrum_equally_spaced(self):
        e = self._levels(self._reconstruct(lc().model()))
        np.testing.assert_allclose(np.diff(e), [2.0, 2.0, 2.0], atol=1e-6)

    def test_transmon_transition_frequency(self):
        e = self._levels(self._reconstruct(transmon().model()))
        # plasma frequency √(8 E_C E_J) = 2 GHz, reduced by anharmonicity
        self.assertAlmostEqual(e[1] - e[0], 1.9487, places=3)


if __name__ == "__main__":
    unittest.main()
