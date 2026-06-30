"""
utils/units.py

Contains physical constants and unit conversion utilities.
Adapted from SQcircuit.
"""


# Henry units
henry_list: dict[str, float] = {
    'H': 1.0, 
    'mH': 1.0e-3, 
    'uH': 1.0e-6,
    'nH': 1.0e-9, 
    'pH': 1.0e-12, 
    'fH': 1.0e-15
}

# Farad units
farad_list: dict[str, float] = {
    'F': 1.0, 
    'mF': 1.0e-3, 
    'uF': 1.0e-6,
    'nF': 1.0e-9, 
    'pF': 1.0e-12, 
    'fF': 1.0e-15
}

# Frequency units
freq_list: dict[str, float] = {
    'Hz': 1.0, 
    'kHz': 1.0e3, 
    'MHz': 1.0e6,
    'GHz': 1.0e9, 
    'THz': 1.0e12
}

# Time units
time_list: dict[str, float] = {
    's': 1.0, 
    'ms': 1.0e-3, 
    'us': 1.0e-6,
    'ns': 1.0e-9, 
    'ps': 1.0e-12, 
    'fs': 1.0e-15
}

# Physical Constants
hbar: float = 1.0545718e-34    # Reduced Planck constant (J·s)
Phi0: float = 2.067833e-15     # Magnetic flux quantum (Wb)
e: float = 1.6021766e-19       # Electron charge (C)
k_B: float = 1.380649e-23      # Boltzmann constant (J/K)

# Global default units configuration
_unit_freq: float = freq_list["GHz"]  # Main frequency unit
_unit_cap: str = "GHz"                # Default capacitor unit
_unit_ind: str = "GHz"                # Default inductor unit
_unit_JJ: str = "GHz"                 # Default JJ unit


def set_unit_freq(unit: str) -> None:
    """
    Change the main frequency unit of the QuantumSCC package.

    Parameters
    ----------
    unit: str
        The desired frequency unit (e.g., "THz", "GHz").
    """
    assert unit in freq_list, "The input format is not correct. Must be in freq_list."
    global _unit_freq
    _unit_freq = freq_list[unit]


def get_unit_freq() -> float:
    """
    Get current frequency unit value in Hertz.
    """
    return _unit_freq


def set_unit_cap(unit: str) -> None:
    """
    Change the default unit for capacitors.

    Parameters
    ----------
    unit: str
        The desired capacitor default unit ("GHz", "nF", etc.).
    """
    if unit not in freq_list and unit not in farad_list:
        raise ValueError(
            "The input unit is not correct. Use a frequency or farad unit."
        )
    global _unit_cap
    _unit_cap = unit


def get_unit_cap() -> str:
    """Get current default unit string for capacitors."""
    return _unit_cap


def set_unit_ind(unit: str) -> None:
    """
    Change the default unit for inductors.

    Parameters
    ----------
    unit: str
        The desired inductor default unit ("GHz", "nH", etc.).
    """
    if unit not in freq_list and unit not in henry_list:
        raise ValueError(
            "The input unit is not correct. Use a frequency or henry unit."
        )
    global _unit_ind
    _unit_ind = unit


def get_unit_ind() -> str:
    """Get current default unit string for inductors."""
    return _unit_ind


def set_unit_JJ(unit: str) -> None:
    """
    Change the default unit for Josephson junctions.

    Parameters
    ----------
    unit: str
        The desired Josephson junction default unit ("GHz", "THz", etc.).
    """
    assert unit in freq_list, "The input format is not correct."
    global _unit_JJ
    _unit_JJ = unit


def get_unit_JJ() -> str:
    """Get current default unit string for Josephson junctions."""
    return _unit_JJ