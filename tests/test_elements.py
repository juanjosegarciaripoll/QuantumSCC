"""
Unit tests for QuantumSCC.core.elements (element construction and validation).

Covers element construction, unit conversion, and error handling for all four
element types: Capacitor, Inductor, Junction, PhaseSlip.
No Circuit object is built here — these are pure element-layer tests.
"""

import unittest
import numpy as np

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

    def test_value_ghz_returns_1(self):
        """Junction(1 GHz).value() == 1.0."""
        self.assertAlmostEqual(Junction(1.0, unit='GHz').value(), 1.0, places=10)

    def test_value_thz_converts_to_ghz(self):
        """Junction(1 THz).value() == 1000.0 GHz."""
        self.assertAlmostEqual(Junction(1.0, unit='THz').value(), 1000.0, places=7)

    def test_value_mhz_converts_to_ghz(self):
        """Junction(1000 MHz).value() == 1.0 GHz."""
        self.assertAlmostEqual(Junction(1000.0, unit='MHz').value(), 1.0, places=8)

    def test_bare_junction_creates(self):
        """Junction without cap parameter creates successfully."""
        j = Junction(1, unit='GHz')
        self.assertAlmostEqual(j.value(), 1.0, places=10)

    def test_invalid_unit_raises(self):
        """Junction with a non-frequency unit raises ValueError."""
        with self.assertRaises(ValueError):
            Junction(1, unit='nH')


# ── PhaseSlip ─────────────────────────────────────────────────────────────────

class TestPhaseSlipConversions(unittest.TestCase):

    def test_value_ghz_returns_1(self):
        """PhaseSlip(1 GHz).value() == 1.0."""
        self.assertAlmostEqual(PhaseSlip(1.0, unit='GHz').value(), 1.0, places=10)

    def test_value_thz_converts_to_ghz(self):
        """PhaseSlip(1 THz).value() == 1000.0 GHz."""
        self.assertAlmostEqual(PhaseSlip(1.0, unit='THz').value(), 1000.0, places=7)

    def test_value_mhz_converts_to_ghz(self):
        """PhaseSlip(1000 MHz).value() == 1.0 GHz."""
        self.assertAlmostEqual(PhaseSlip(1000.0, unit='MHz').value(), 1.0, places=8)

    def test_bare_phaseslip_creates(self):
        """PhaseSlip without L_value parameter creates successfully."""
        p = PhaseSlip(1, unit='GHz')
        self.assertAlmostEqual(p.value(), 1.0, places=10)

    def test_invalid_unit_nH_raises(self):
        """PhaseSlip with an inductance unit raises ValueError."""
        with self.assertRaises(ValueError):
            PhaseSlip(1, unit='nH')

    def test_invalid_unit_pF_raises(self):
        """PhaseSlip with a capacitance unit raises ValueError."""
        with self.assertRaises(ValueError):
            PhaseSlip(1, unit='pF')


# ── Positive energy validation ────────────────────────────────────────────────

class TestPositiveEnergyValidation(unittest.TestCase):
    """All elements must reject zero and negative energy/value."""

    def test_capacitor_negative_raises(self):
        with self.assertRaises(ValueError):
            Capacitor(-1, 'GHz')

    def test_capacitor_zero_raises(self):
        with self.assertRaises(ValueError):
            Capacitor(0, 'GHz')

    def test_capacitor_negative_farads_raises(self):
        with self.assertRaises(ValueError):
            Capacitor(-1, 'pF')

    def test_inductor_negative_raises(self):
        with self.assertRaises(ValueError):
            Inductor(-1, 'GHz')

    def test_inductor_zero_raises(self):
        with self.assertRaises(ValueError):
            Inductor(0, 'nH')

    def test_junction_negative_raises(self):
        with self.assertRaises(ValueError):
            Junction(-5, 'GHz')

    def test_junction_zero_raises(self):
        with self.assertRaises(ValueError):
            Junction(0, 'GHz')

    def test_phaseslip_negative_raises(self):
        with self.assertRaises(ValueError):
            PhaseSlip(-1, 'GHz')

    def test_phaseslip_zero_raises(self):
        with self.assertRaises(ValueError):
            PhaseSlip(0, 'GHz')

    def test_positive_values_accepted(self):
        """Positive values should work fine (sanity check)."""
        Capacitor(1, 'GHz')
        Inductor(1, 'GHz')
        Junction(1, 'GHz')
        PhaseSlip(1, 'GHz')


if __name__ == '__main__':
    unittest.main()
