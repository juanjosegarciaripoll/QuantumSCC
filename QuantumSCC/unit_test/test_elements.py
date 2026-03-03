"""
Unit tests for QuantumSCC.core.elements (element construction and validation).

Covers element construction, unit conversion, and error handling for all four
element types: Capacitor, Inductor, Junction, PhaseSlip.
No Circuit object is built here — these are pure element-layer tests.
"""

import unittest
import numpy as np
import os
import sys

current_dir  = os.path.dirname(os.path.abspath(__file__))
package_dir  = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC.utils import units as unt
from QuantumSCC.core.elements import Capacitor, Inductor, Junction, PhaseSlip


# ── Capacitor ─────────────────────────────────────────────────────────────────

class TestCapacitorConversions(unittest.TestCase):

    def test_value_pF_to_F(self):
        """Capacitor(1 pF).value() == 1e-12 F."""
        self.assertAlmostEqual(Capacitor(1, unit='pF').value(), 1e-12, places=24)

    def test_value_nF_to_F(self):
        """Capacitor(1 nF).value() == 1e-9 F."""
        self.assertAlmostEqual(Capacitor(1, unit='nF').value(), 1e-9, places=21)

    def test_value_fF_to_F(self):
        """Capacitor(1 fF).value() == 1e-15 F."""
        self.assertAlmostEqual(Capacitor(1, unit='fF').value(), 1e-15, places=27)

    def test_energy_ghz_unit_returns_1(self):
        """Capacitor(1 GHz).energy() == 1.0 (already in GHz)."""
        self.assertAlmostEqual(Capacitor(1.0, unit='GHz').energy(), 1.0, places=10)

    def test_energy_from_farads_formula(self):
        """E_C = (2e)² / (2C·ℏ), consistent with physical formula."""
        C_F = 1e-12
        cap = Capacitor(1, unit='pF')
        E_c_theory = (2 * unt.e)**2 / (2 * C_F) / unt.hbar / unt.freq_list['GHz']
        self.assertAlmostEqual(cap.energy(), E_c_theory, places=5)

    def test_energy_proportional_to_inverse_capacitance(self):
        """Doubling C halves E_C."""
        e1 = Capacitor(1, unit='pF').energy()
        e2 = Capacitor(2, unit='pF').energy()
        self.assertAlmostEqual(e1 / e2, 2.0, places=10)

    def test_invalid_unit_raises(self):
        with self.assertRaises(ValueError):
            Capacitor(1, unit='km')


# ── Inductor ──────────────────────────────────────────────────────────────────

class TestInductorConversions(unittest.TestCase):

    def test_value_nH_to_H(self):
        """Inductor(1 nH).value() == 1e-9 H."""
        self.assertAlmostEqual(Inductor(1, unit='nH').value(), 1e-9, places=21)

    def test_value_pH_to_H(self):
        """Inductor(1 pH).value() == 1e-12 H."""
        self.assertAlmostEqual(Inductor(1, unit='pH').value(), 1e-12, places=24)

    def test_energy_ghz_unit_returns_1(self):
        """Inductor(1 GHz).energy() == 1.0 (already in GHz)."""
        self.assertAlmostEqual(Inductor(1.0, unit='GHz').energy(), 1.0, places=10)

    def test_energy_from_henries_formula(self):
        """E_L = (Φ₀/2π)² / (2L·ℏ), consistent with physical formula."""
        L_H = 1e-9
        ind = Inductor(1, unit='nH')
        E_l_theory = (unt.Phi0 / (2 * np.pi))**2 / (2 * L_H) / unt.hbar / unt.freq_list['GHz']
        self.assertAlmostEqual(ind.energy(), E_l_theory, places=5)

    def test_energy_proportional_to_inverse_inductance(self):
        """Doubling L halves E_L."""
        e1 = Inductor(1, unit='nH').energy()
        e2 = Inductor(2, unit='nH').energy()
        self.assertAlmostEqual(e1 / e2, 2.0, places=10)

    def test_invalid_unit_raises(self):
        with self.assertRaises(ValueError):
            Inductor(1, unit='km')


# ── Junction ──────────────────────────────────────────────────────────────────

class TestJunctionConversions(unittest.TestCase):

    def _cap(self):
        return Capacitor(1, unit='pF')

    def test_value_ghz_returns_1(self):
        """Junction(1 GHz).value() == 1.0."""
        self.assertAlmostEqual(Junction(1.0, unit='GHz', cap=self._cap()).value(), 1.0, places=10)

    def test_value_thz_converts_to_ghz(self):
        """Junction(1 THz).value() == 1000.0 GHz."""
        self.assertAlmostEqual(Junction(1.0, unit='THz', cap=self._cap()).value(), 1000.0, places=7)

    def test_value_mhz_converts_to_ghz(self):
        """Junction(1000 MHz).value() == 1.0 GHz."""
        self.assertAlmostEqual(Junction(1000.0, unit='MHz', cap=self._cap()).value(), 1.0, places=8)

    def test_missing_cap_raises(self):
        """Junction without a parallel capacitor raises ValueError."""
        with self.assertRaises(ValueError):
            Junction(1, unit='GHz')

    def test_invalid_unit_raises(self):
        """Junction with a non-frequency unit raises ValueError."""
        with self.assertRaises(ValueError):
            Junction(1, unit='nH', cap=self._cap())

    def test_parallel_cap_stored(self):
        """cap attribute must be the capacitor passed at construction."""
        cap = self._cap()
        j = Junction(1.0, unit='GHz', cap=cap)
        self.assertIs(j.cap, cap)


# ── PhaseSlip ─────────────────────────────────────────────────────────────────

class TestPhaseSlipConversions(unittest.TestCase):

    def _ind(self):
        return Inductor(1, unit='nH')

    def test_value_ghz_returns_1(self):
        """PhaseSlip(1 GHz).value() == 1.0."""
        self.assertAlmostEqual(PhaseSlip(1.0, unit='GHz', ind=self._ind()).value(), 1.0, places=10)

    def test_value_thz_converts_to_ghz(self):
        """PhaseSlip(1 THz).value() == 1000.0 GHz."""
        self.assertAlmostEqual(PhaseSlip(1.0, unit='THz', ind=self._ind()).value(), 1000.0, places=7)

    def test_value_mhz_converts_to_ghz(self):
        """PhaseSlip(1000 MHz).value() == 1.0 GHz."""
        self.assertAlmostEqual(PhaseSlip(1000.0, unit='MHz', ind=self._ind()).value(), 1.0, places=8)

    def test_missing_ind_raises(self):
        """PhaseSlip without a parallel inductor raises ValueError."""
        with self.assertRaises(ValueError):
            PhaseSlip(1, unit='GHz')

    def test_invalid_unit_nH_raises(self):
        """PhaseSlip with an inductance unit raises ValueError."""
        with self.assertRaises(ValueError):
            PhaseSlip(1, unit='nH', ind=self._ind())

    def test_invalid_unit_pF_raises(self):
        """PhaseSlip with a capacitance unit raises ValueError."""
        with self.assertRaises(ValueError):
            PhaseSlip(1, unit='pF', ind=self._ind())

    def test_parallel_inductor_stored(self):
        """ind attribute must be the inductor passed at construction."""
        ind = self._ind()
        ps = PhaseSlip(1.0, unit='GHz', ind=ind)
        self.assertIs(ps.ind, ind)


if __name__ == '__main__':
    unittest.main()
