"""
core/elements.py

Contains the classes for the circuit elements:
Capacitors, Inductors, and Josephson Junctions.
"""

from typing import Optional, Union
import numpy as np
from ..utils import units as unt

class Capacitor:
    """
    Class that contains the capacitor properties.

    Parameters
    ----------
    value: float
        The value of the capacitor.
    unit: Optional[str]
        The unit of input value. 
        - If unit is "THz", "GHz", etc., the value specifies charging energy.
        - If unit is "fF", "pF", etc., the value specifies capacitance in Farads.
        - If None, uses the default unit from units module.
    """

    def __init__(
        self,
        value: float,
        unit: Optional[str] = None,
    ) -> None:

        if (
            unit not in unt.freq_list
            and unit not in unt.farad_list
            and unit is not None
        ):
            raise ValueError(
                "The input unit for the capacitor is not correct. "
                "Look at the documentation for the correct input format."
            )

        self.cValue = value
        self.type = type(self)

        if unit is None:
            self.unit = unt.get_unit_cap()
        else:
            self.unit = unit

    def value(self, random: bool = False) -> float:
        """
        Return the value of the capacitor in Farad units.
        """
        if self.unit in unt.farad_list:
            cMean = self.cValue * unt.farad_list[self.unit]
        else:
            # E_c to Capacitance conversion
            E_c = self.cValue * unt.freq_list[self.unit] * unt.hbar
            cMean = (2 * unt.e)**2 / (2 * E_c)
            
        if not random:
            return cMean
        else:
            # Simple fabrication error model
            return np.random.normal(cMean, cMean * getattr(self, 'error', 0) / 100, 1)[0]

    def energy(self) -> float:
        """
        Return the charging energy of the capacitor in the main frequency unit 
        of SQcircuit (GHz by default).
        """
        if self.unit in unt.freq_list:
            return self.cValue * unt.freq_list[self.unit] / unt.get_unit_freq()
        else:
            c = self.cValue * unt.farad_list[self.unit]
            # Capacitance to Energy conversion
            return ((2 * unt.e)**2 / (2 * c)) / unt.hbar / unt.get_unit_freq()


class Inductor:
    """
    Class that contains the inductor properties.

    Parameters
    ----------
    value: float
        The value of the inductor.
    unit: Optional[str]
        The unit of input value.
        - If unit is "THz", "GHz", etc., the value specifies inductive energy.
        - If unit is "fH", "pH", etc., the value specifies inductance in Henry.
        - If None, uses the default unit from units module.
    """

    def __init__(
        self,
        value: float,
        unit: Optional[str] = None,
    ) -> None:

        if (
            unit not in unt.freq_list
            and unit not in unt.henry_list
            and unit is not None
        ):
            raise ValueError(
                "The input unit for the inductor is not correct. "
                "Look at the documentation for the correct input format."
            )

        self.lValue = value
        self.type = type(self)

        if unit is None:
            self.unit = unt.get_unit_ind()
        else:
            self.unit = unit

    def value(self, random: bool = False) -> float:
        """
        Return the value of the inductor in Henry units.
        """
        if self.unit in unt.henry_list:
            lMean = self.lValue * unt.henry_list[self.unit]
        else:
            # Energy to Inductance conversion
            E_l = self.lValue * unt.freq_list[self.unit] * unt.hbar
            lMean = (unt.Phi0 / (2 * np.pi)) ** 2 / (2 * E_l)

        if not random:
            return lMean
        else:
            return np.random.normal(lMean, lMean * getattr(self, 'error', 0) / 100, 1)[0]

    def energy(self) -> float:
        """
        Return the inductive energy of the inductor in the main frequency unit 
        of SQcircuit (GHz by default).
        """
        if self.unit in unt.freq_list:
            return self.lValue * unt.freq_list[self.unit] / unt.get_unit_freq()
        else:
            l = self.lValue * unt.henry_list[self.unit]
            # Inductance to Energy conversion
            return (
                (unt.Phi0 / (2 * np.pi)) ** 2
                / (2 * l)
                / unt.hbar
                / unt.get_unit_freq()
            )


class Junction:
    """
    Class that contains the Josephson junction properties.

    Single-branch element: carries only the nonlinear energy -E_J cos(phi).
    The user must add a parallel Capacitor explicitly if needed (e.g. for transmon).

    Parameters
    -----------
    value: float
        The Josephson energy E_J.
    unit: Optional[str]
        The unit of input value ("THz", "GHz", etc.).
    """

    def __init__(
        self,
        value: float,
        unit: Optional[str] = None,
    ) -> None:

        if unit not in unt.freq_list and unit is not None:
            raise ValueError(
                "The input unit for the Josephson Junction is not correct."
            )

        self.jValue = value
        self.type = type(self)

        if unit is None:
            self.unit = unt.get_unit_JJ()
        else:
            self.unit = unit

    def value(self) -> float:
        """
        Return the value of the Josephson Junction in the main frequency unit (GHz).
        """
        jMean = self.jValue * unt.freq_list[self.unit] / unt.get_unit_freq()
        return jMean


class PhaseSlip:
    """
    Quantum Phase Slip (QPS) element — electromagnetic dual of Josephson Junction.

    Single-branch element: carries only the nonlinear energy -E_P cos(pi q / e).
    The user must add a parallel Inductor explicitly if needed (e.g. for dual-transmon).

    Parameters
    ----------
    value : float
        Phase slip amplitude E_P in frequency units.
    unit : Optional[str]
        Frequency unit (GHz, THz, etc.).
    """

    def __init__(
        self,
        value: float,
        unit: Optional[str] = None,
    ) -> None:

        if unit not in unt.freq_list and unit is not None:
            raise ValueError(
                "The input unit for the PhaseSlip element is not correct. "
                "It must be a frequency unit (GHz, THz, ...)."
            )

        self.pValue = value
        self.type = type(self)

        if unit is None:
            self.unit = unt.get_unit_JJ()
        else:
            self.unit = unit

    def value(self) -> float:
        """
        Return E_P in the main frequency unit (GHz by default).
        """
        pMean = self.pValue * unt.freq_list[self.unit] / unt.get_unit_freq()
        return pMean