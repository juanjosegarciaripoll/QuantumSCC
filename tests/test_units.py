"""
Unit tests for QuantumSCC.utils.units.

Covers:
  - Unit conversion dictionaries (farad_list, henry_list, freq_list)
  - Physical constants (hbar, Phi0, e, k_B)
  - Global set_unit_*/get_unit_* helper functions

Element construction and unit conversion for Capacitor / Inductor /
Junction / PhaseSlip are tested in test_elements.py.
"""

import unittest

import numpy as np

from QuantumSCC.utils import units as unt

# ── 1. Unit conversion tables ─────────────────────────────────────────────────

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
        for a, b in zip(prefixes, prefixes[1:], strict=False):
            self.assertAlmostEqual(unt.farad_list[b] / unt.farad_list[a], 1e-3, places=20)

    def test_henry_consecutive_ratio(self):
        prefixes = ['H', 'mH', 'uH', 'nH', 'pH', 'fH']
        for a, b in zip(prefixes, prefixes[1:], strict=False):
            self.assertAlmostEqual(unt.henry_list[b] / unt.henry_list[a], 1e-3, places=20)


# ── 2. Physical constants ─────────────────────────────────────────────────────

class TestPhysicalConstants(unittest.TestCase):

    def test_hbar_value(self):
        """hbar == 1.0545718e-34 J·s."""
        self.assertEqual(unt.hbar, 1.0545718e-34)

    def test_electron_charge_value(self):
        """e == 1.6021766e-19 C."""
        self.assertEqual(unt.e, 1.6021766e-19)

    def test_flux_quantum_consistent(self):
        """Phi0 ≈ h / (2e) = 2π·ℏ / (2e), within 0.01 % of stored value."""
        Phi0_theory = 2 * np.pi * unt.hbar / (2 * unt.e)
        self.assertAlmostEqual(unt.Phi0 / Phi0_theory, 1.0, places=3)

    def test_boltzmann_constant(self):
        """k_B ≈ 1.381e-23 J/K."""
        self.assertAlmostEqual(unt.k_B, 1.380649e-23, places=30)


# ── 3. Global unit set / get functions ───────────────────────────────────────

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
