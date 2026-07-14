"""
QuantumSCC.diag — exact-diagonalisation solver for QuantumSCC circuit models.

Consumes a :class:`QuantumSCC.CircuitModel` (or a solved :class:`QuantumSCC.Circuit`)
and builds a ``scipy.sparse`` Hamiltonian, then diagonalises it. Each mode is
placed in a finite Hilbert space chosen from its sector — charge basis for the
compact (Josephson / phase-slip) modes, harmonic-oscillator Fock basis for the
extended modes — following the recipe in ``docs/model_extraction.md``.

This is an explicit submodule: import what you need from it directly, e.g.

    from QuantumSCC.diag import eigenenergies, build_hamiltonian
"""

from .builder import build_hamiltonian, eigenenergies, eigenstates
from .operators import (
    Mode,
    annihilation,
    charge_mode,
    displacement,
    dual_charge_mode,
    oscillator_mode,
)

__all__ = [
    "build_hamiltonian",
    "eigenenergies",
    "eigenstates",
    "Mode",
    "oscillator_mode",
    "charge_mode",
    "dual_charge_mode",
    "displacement",
    "annihilation",
]
