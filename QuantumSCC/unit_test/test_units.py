"""
Unit tests for QuantumSCC.utils.units and element conversions.

Covers:
  - Unit conversion dictionaries (farad_list, henry_list, freq_list)
  - Physical constants (hbar, Phi0, e)
  - Capacitor.value() and Capacitor.energy() for different input units
  - Inductor.value()  and Inductor.energy()  for different input units
  - Junction.value() and error handling
  - set_unit_*/get_unit_* helper functions
"""

import unittest
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(package_dir)
sys.path.insert(0, project_root)

from QuantumSCC.utils import units as unt
from QuantumSCC.core.elements import Capacitor, Inductor, Junction, PhaseSlip


# ---------------------------------------------------------------------------
# 1. Unit conversion tables
# ---------------------------------------------------------------------------

class TestUnitTables(unittest.TestCase):
    """Verify the SI prefix conversion factors."""

    def test_farad_prefix_values(self):
        self.assertEqual(unt.farad_list['F'],   1.0)
        self.assertAlmostEqual(unt.farad_list['mF'],  1e-3)
        self.assertAlmostEqual(unt.farad_list['uF'],  1e-6)
        self.assertAlmostEqual(unt.farad_list['nF'],  1e-9)
        self.assertAlmostEqual(unt.farad_list['pF'],  1e-12)
        self.assertAlmostEqual(unt.farad_list['fF'],  1e-15)

    def test_henry_prefix_values(self):
        self.assertEqual(unt.henry_list['H'],   1.0)
        self.assertAlmostEqual(unt.henry_list['mH'],  1e-3)
        self.assertAlmostEqual(unt.henry_list['uH'],  1e-6)
        self.assertAlmostEqual(unt.henry_list['nH'],  1e-9)
        self.assertAlmostEqual(unt.henry_list['pH'],  1e-12)
        self.assertAlmostEqual(unt.henry_list['fH'],  1e-15)

    def test_freq_prefix_values(self):
        self.assertEqual(unt.freq_list['Hz'],  1.0)
        self.assertAlmostEqual(unt.freq_list['kHz'],  1e3)
        self.assertAlmostEqual(unt.freq_list['MHz'],  1e6)
        self.assertAlmostEqual(unt.freq_list['GHz'],  1e9)
        self.assertAlmostEqual(unt.freq_list['THz'],  1e12)

    def test_farad_consecutive_ratio(self):
        """Each Farad prefix is exactly 1e-3 times the previous one."""
        prefixes = ['F', 'mF', 'uF', 'nF', 'pF', 'fF']
        for a, b in zip(prefixes, prefixes[1:]):
            self.assertAlmostEqual(unt.farad_list[b] / unt.farad_list[a], 1e-3, places=20)

    def test_henry_consecutive_ratio(self):
        prefixes = ['H', 'mH', 'uH', 'nH', 'pH', 'fH']
        for a, b in zip(prefixes, prefixes[1:]):
            self.assertAlmostEqual(unt.henry_list[b] / unt.henry_list[a], 1e-3, places=20)


# ---------------------------------------------------------------------------
# 2. Physical constants
# ---------------------------------------------------------------------------

class TestPhysicalConstants(unittest.TestCase):

    def test_hbar_value(self):
        """hbar == 1.0545718e-34 J·s (stored as a float literal)."""
        self.assertEqual(unt.hbar, 1.0545718e-34)

    def test_electron_charge_value(self):
        """e == 1.6021766e-19 C."""
        self.assertEqual(unt.e, 1.6021766e-19)

    def test_flux_quantum_consistent(self):
        """Phi0 ≈ h / (2e) = 2π*hbar / (2e), within 0.01% of stored value."""
        Phi0_theory = 2 * np.pi * unt.hbar / (2 * unt.e)
        self.assertAlmostEqual(unt.Phi0 / Phi0_theory, 1.0, places=3)

    def test_boltzmann_constant(self):
        """k_B ≈ 1.381e-23 J/K."""
        self.assertAlmostEqual(unt.k_B, 1.380649e-23, places=30)


# ---------------------------------------------------------------------------
# 3. Capacitor conversions
# ---------------------------------------------------------------------------

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
        """E_C = (2e)² / (2C) / hbar / GHz, consistent to 5 decimal places."""
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


# ---------------------------------------------------------------------------
# 4. Inductor conversions
# ---------------------------------------------------------------------------

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
        """E_L = (Phi0/2π)² / (2L) / hbar / GHz, consistent to 5 decimal places."""
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


# ---------------------------------------------------------------------------
# 5. Junction conversions and validation
# ---------------------------------------------------------------------------

class TestJunctionConversions(unittest.TestCase):

    def _cap(self):
        return Capacitor(1, unit='pF')

    def test_value_ghz_returns_1(self):
        """Junction(1 GHz).value() == 1.0."""
        self.assertAlmostEqual(Junction(1.0, unit='GHz', cap=self._cap()).value(), 1.0, places=10)

    def test_value_thz_converts_to_ghz(self):
        """Junction(1 THz).value() == 1000.0 GHz."""
        self.assertAlmostEqual(Junction(1.0, unit='THz', cap=self._cap()).value(), 1000.0, places=7)

    def test_missing_cap_raises(self):
        """Junction without a parallel capacitor raises ValueError."""
        with self.assertRaises(ValueError):
            Junction(1, unit='GHz')

    def test_invalid_unit_raises(self):
        """Junction with a non-frequency unit raises ValueError."""
        with self.assertRaises(ValueError):
            Junction(1, unit='nH', cap=self._cap())


# ---------------------------------------------------------------------------
# 6. PhaseSlip conversions and validation
# ---------------------------------------------------------------------------

class TestPhaseSlipConversions(unittest.TestCase):
    """Tests for PhaseSlip element — dual of Junction."""

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

    def test_invalid_unit_raises(self):
        """PhaseSlip with a non-frequency unit raises ValueError."""
        with self.assertRaises(ValueError):
            PhaseSlip(1, unit='nH', ind=self._ind())

    def test_invalid_unit_pf_raises(self):
        """PhaseSlip with pF unit raises ValueError."""
        with self.assertRaises(ValueError):
            PhaseSlip(1, unit='pF', ind=self._ind())

    def test_parallel_inductor_stored(self):
        """ind attribute must be the inductor passed at construction."""
        ind = self._ind()
        ps = PhaseSlip(1.0, unit='GHz', ind=ind)
        self.assertIs(ps.ind, ind)


# ---------------------------------------------------------------------------
# 7. Global unit set/get functions
# ---------------------------------------------------------------------------

class TestSetGetFunctions(unittest.TestCase):
    """All tests restore global state via tearDown."""

    def setUp(self):
        self._orig_freq = unt.get_unit_freq()
        self._orig_cap  = unt.get_unit_cap()
        self._orig_ind  = unt.get_unit_ind()
        self._orig_jj   = unt.get_unit_JJ()

    def tearDown(self):
        unt._unit_freq = self._orig_freq
        unt._unit_cap  = self._orig_cap
        unt._unit_ind  = self._orig_ind
        unt._unit_JJ   = self._orig_jj

    def test_default_freq_is_ghz(self):
        self.assertEqual(unt.get_unit_freq(), 1e9)

    def test_set_unit_freq_thz(self):
        unt.set_unit_freq('THz')
        self.assertEqual(unt.get_unit_freq(), 1e12)

    def test_set_unit_freq_invalid_raises(self):
        with self.assertRaises(AssertionError):
            unt.set_unit_freq('parsec')

    def test_set_unit_cap_pf(self):
        unt.set_unit_cap('pF')
        self.assertEqual(unt.get_unit_cap(), 'pF')

    def test_set_unit_cap_invalid_raises(self):
        with self.assertRaises(ValueError):
            unt.set_unit_cap('lightyear')

    def test_set_unit_ind_nh(self):
        unt.set_unit_ind('nH')
        self.assertEqual(unt.get_unit_ind(), 'nH')

    def test_set_unit_ind_invalid_raises(self):
        with self.assertRaises(ValueError):
            unt.set_unit_ind('lightyear')

    def test_set_unit_jj_thz(self):
        unt.set_unit_JJ('THz')
        self.assertEqual(unt.get_unit_JJ(), 'THz')

    def test_set_unit_jj_invalid_raises(self):
        with self.assertRaises(AssertionError):
            unt.set_unit_JJ('pF')


if __name__ == '__main__':
    unittest.main()
