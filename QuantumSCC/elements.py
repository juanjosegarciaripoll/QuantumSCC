"""
elements.py contains the classes for the circuit elements:
capacitors, inductors, and josephson junctions.

This file was adapted from SQcircuit 
"""

from typing import List, Any, Optional, Union, Callable

import numpy as np

from scipy.special import kn

from . import units as unt


class Capacitor:
    """
    Class that contains the capacitor properties.

    Parameters
    ----------
    value:
        The value of the capacitor.
    unit:
        The unit of input value. If ``unit`` is "THz", "GHz", and ,etc.,
        the value specifies the charging energy of the capacitor. If ``unit``
        is "fF", "pF", and ,etc., the value specifies the capacitance in
        farad. If ``unit`` is ``None``, the default unit of capacitor is "GHz".
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
            error = (
                "The input unit for the capacitor is not correct. "
                "Look at the documentation for the correct input format."
            )
            raise ValueError(error)

        self.cValue = value
        self.type = type(self)

        if unit is None:
            self.unit = unt.get_unit_cap()
        else:
            self.unit = unit

    def value(self, random: bool = False) -> float:
        """
        Return the value of the capacitor in farad units. If `random` is
        `True`, it samples from a normal distribution with variance defined
        by the fabrication error.

        Parameters
        ----------
            random:
                A boolean flag which specifies whether the output is
                deterministic or random.
        """
        if self.unit in unt.farad_list:
            cMean = self.cValue * unt.farad_list[self.unit]
        else:
            E_c = self.cValue * unt.freq_list[self.unit] * (unt.hbar)   # E_c = self.cValue * unt.freq_list[self.unit] * (2 * np.pi * unt.hbar)
            cMean = (2*unt.e)**2 / 2 / E_c 
        if not random:
            return cMean
        else:
            return np.random.normal(cMean, cMean * self.error / 100, 1)[0]

    def energy(self) -> float:
        """
        Return the charging energy of the capacitor in frequency unit of
        SQcircuit (gigahertz by default).
        """
        if self.unit in unt.freq_list:
            return self.cValue * unt.freq_list[self.unit] / unt.get_unit_freq()
        else:
            c = self.cValue * unt.farad_list[self.unit]
            return (2*unt.e)**2 / 2 / c / (unt.hbar) / unt.get_unit_freq()   # (2*unt.e)**2 / 2 / c / (2 * np.pi * unt.hbar) / unt.get_unit_freq()


class Inductor:
    """
    Class that contains the inductor properties.

    Parameters
    ----------
    value:
        The value of the inductor.
    unit:
        The unit of input value. If ``unit`` is "THz", "GHz", and ,etc.,
        the value specifies the inductive energy of the inductor. If ``unit``
        is "fH", "pH", and ,etc., the value specifies the inductance in henry.
        If ``unit`` is ``None``, the default unit of inductor is "GHz".
    """

    def __init__(
        self,
        value: float,
        unit: str = None,
    ) -> None:

        if (
            unit not in unt.freq_list
            and unit not in unt.henry_list
            and unit is not None
        ):
            error = (
                "The input unit for the inductor is not correct. "
                "Look at the documentation for the correct input format."
            )
            raise ValueError(error)

        self.lValue = value
        self.type = type(self)


        if unit is None:
            self.unit = unt.get_unit_ind()
        else:
            self.unit = unit

    def value(self, random: bool = False) -> float:
        """
        Return the value of the inductor in henry units. If `random` is
        `True`, it samples from a normal distribution with variance defined
        by the fabrication error.

        Parameters
        ----------
            random:
                A boolean flag which specifies whether the output is
                deterministic or random.
        """
        if self.unit in unt.henry_list:
            lMean = self.lValue * unt.henry_list[self.unit]
        else:
            E_l = self.lValue * unt.freq_list[self.unit] * (unt.hbar)  # E_l = self.lValue * unt.freq_list[self.unit] * (2 * np.pi * unt.hbar)
            lMean = (unt.Phi0/2/np.pi) ** 2 / (2 * E_l)  #(unt.Phi0) ** 2 / (2 * E_l)

        if not random:
            return lMean
        else:
            return np.random.normal(lMean, lMean * self.error / 100, 1)[0]

    def energy(self) -> float:
        """
        Return the inductive energy of the capacitor in frequency unit of
        SQcircuit (gigahertz by default).
        """
        if self.unit in unt.freq_list:
            return self.lValue * unt.freq_list[self.unit] / unt.get_unit_freq()
        else:
            l = self.lValue * unt.henry_list[self.unit]
            return (
                (unt.Phi0/2/np.pi) ** 2  #(unt.Phi0) ** 2
                / (2 * l)
                / (unt.hbar) # (2 * np.pi * unt.hbar) 
                / unt.get_unit_freq()
            )


class Junction:
    """
    Class that contains the Josephson junction properties.

    Parameters
    -----------
    value:
        The value of the Josephson junction.
    unit: str
        The unit of input value. The ``unit`` can be "THz", "GHz", and ,etc.,
        that specifies the junction energy of the inductor. If ``unit`` is
        ``None``, the default unit of junction is "GHz".
    cap:
        Capacitor associated to the josephson junction, necessary for the
        correct correct operation of the program.
    """

    def __init__(
        self,
        value: float,
        unit: Optional[str] = None,
        cap: Optional[str] = None,
    ) -> None:

        if unit not in unt.freq_list and unit is not None:
            error = (
                "The input unit for the Josephson Junction is not "
                "correct. Look at the documentation for the correct "
                "input format."
            )
            raise ValueError(error)

        self.jValue = value
        self.type = type(self)

        if unit is None:
            self.unit = unt.get_unit_JJ()
        else:
            self.unit = unit

        if cap is None:
            raise ValueError("For the correct operation of the program, each Josephson junction must have a parallel capacitor.")
        else:
            self.cap = cap

    def value(self) -> float:
        """
        Return the value of the Josephson Junction in GHz.

        """
        jMean = self.jValue * unt.freq_list[self.unit]/ unt.get_unit_freq() #* 2 * np.pi
        #jMean = self.jValue * unt.freq_list[self.unit] * 2 * np.pi

        return jMean
       