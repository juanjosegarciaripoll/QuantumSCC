"""
Sparse single-mode operators for the two representations the solver uses.

Each mode of a :class:`QuantumSCC.CircuitModel` is placed in a finite Hilbert
space chosen from its sector (see ``QuantumSCC/docs/model_extraction.md``):

- **extended** (oscillator) → harmonic-oscillator / Fock basis. Both the flux
  ``φ`` and charge ``n`` operators are well defined.
- **compact_flux** (Josephson) → charge basis ``|n⟩``, ``n ∈ [−cut, cut]``. The
  charge ``n`` is the well-defined operator; the flux ``φ`` is periodic and only
  ever enters through ``exp(i c φ)`` (a charge-shift operator).
- **compact_charge** (phase-slip / dual) → dual charge basis ``|ψ⟩``. Roles
  swap: the flux ``ψ`` is well defined and the charge ``q`` is periodic.

Every factory returns a :class:`Mode`, which exposes the well-defined operators
(``flux`` / ``charge``, ``None`` when periodic) and two callables building
``exp(i c φ)`` and ``exp(i c n)`` — the factors a cosine decomposes into.

All operators are ``scipy.sparse`` matrices in ``csc`` format. ``scipy`` is
untyped here (see ``[tool.mypy]`` in ``pyproject.toml``), so the ``Sparse``
alias resolves to ``Any``; it is used purely to document intent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

type Sparse = Any

__all__ = [
    "Mode",
    "oscillator_mode",
    "charge_mode",
    "dual_charge_mode",
    "displacement",
    "annihilation",
]


@dataclass(frozen=True)
class Mode:
    """Single-mode operator set on a Hilbert space of dimension ``dim``.

    ``flux`` and ``charge`` are the well-defined (non-periodic) coordinate
    operators, or ``None`` when that coordinate is periodic and may appear only
    inside a cosine. ``exp_flux`` / ``exp_charge`` map a real coefficient ``c``
    to ``exp(i c φ)`` / ``exp(i c n)`` respectively.
    """

    dim: int
    flux: Sparse | None
    charge: Sparse | None
    exp_flux: Callable[[float], Sparse]
    exp_charge: Callable[[float], Sparse]


def annihilation(dim: int) -> Sparse:
    """Bosonic annihilation operator truncated to ``dim`` Fock states."""
    return sp.diags(np.sqrt(np.arange(1, dim)), 1, format="csc")


def _exp_nilpotent(matrix: Sparse, dim: int) -> Sparse:
    """``exp(matrix)`` for a strictly-triangular (nilpotent) ladder operator.

    On the truncated space ``matrix ** dim == 0``, so the Taylor series
    terminates: the result is the exact finite sum ``Σ_k matrix**k / k!`` with no
    Padé approximant and no scaling-and-squaring.
    """
    term = sp.identity(dim, format="csc", dtype=complex)
    result = term
    for k in range(1, dim):
        term = (term @ matrix) / k
        if term.nnz == 0:
            break
        result = result + term
    return result.tocsc()


def displacement(alpha: complex, dim: int) -> Sparse:
    """Displacement operator ``D(α) = exp(α a† − α* a)`` on ``dim`` Fock states.

    Built from the normal-ordered Baker–Campbell–Hausdorff factorisation

        ``D(α) = e^{−|α|²/2} · exp(α a†) · exp(−α* a)``,

    exact because ``[a, a†]`` is central. Each exponential is of a nilpotent
    ladder operator and so is an exact terminating series (see
    :func:`_exp_nilpotent`) — more accurate and better-conditioned than a dense
    ``expm`` of the anti-Hermitian generator ``α a† − α* a``.
    """
    a = annihilation(dim)
    ad = a.conj().T
    raising = _exp_nilpotent(alpha * ad, dim)
    lowering = _exp_nilpotent(-np.conjugate(alpha) * a, dim)
    prefactor = float(np.exp(-0.5 * abs(alpha) ** 2))
    return prefactor * (raising @ lowering)


def oscillator_mode(dim: int) -> Mode:
    """Extended mode in the Fock basis with ``dim`` levels.

    Uses the dimensionless convention ``φ = (a + a†)/√2``, ``n = i(a† − a)/√2``,
    so ``½(φ² + n²) = a†a + ½``. The cosine factors are displacement operators:
    ``exp(i c φ) = D(i c/√2)`` and ``exp(i c n) = D(−c/√2)``.
    """
    a = annihilation(dim)
    ad = a.conj().T
    flux = (a + ad) / np.sqrt(2)
    charge = 1j * (ad - a) / np.sqrt(2)
    return Mode(
        dim=dim,
        flux=flux,
        charge=charge,
        exp_flux=lambda c: displacement(1j * c / np.sqrt(2), dim),
        exp_charge=lambda c: displacement(-c / np.sqrt(2), dim),
    )


def _charge_shift(dim: int) -> Sparse:
    """Ladder operator raising the integer occupation by one (``|k⟩ → |k+1⟩``)."""
    return sp.diags(np.ones(dim - 1), -1, format="csc")


def _shift_power(shift: Sparse, coeff: float) -> Sparse:
    """``shift ** round(coeff)`` for either sign, the periodic ``exp(i c θ)``.

    Coupling vectors over periodic coordinates are integers (a junction adds or
    removes whole Cooper pairs), so ``coeff`` is rounded to the nearest integer.
    """
    power = int(round(coeff))
    if power >= 0:
        return shift**power
    return (shift.conj().T) ** (-power)


def charge_mode(cut: int) -> Mode:
    """Compact-flux (Josephson) mode in the charge basis ``|n⟩``.

    ``n`` runs over the integers ``[−cut, cut]`` (``2·cut + 1`` states). The
    charge is diagonal; the flux is periodic, so ``exp(i c φ)`` is the
    charge-shift operator ``n → n + round(c)``.
    """
    vals = np.arange(-cut, cut + 1)
    dim = 2 * cut + 1
    charge = sp.diags(vals, 0, format="csc", dtype=float)
    shift = _charge_shift(dim)
    return Mode(
        dim=dim,
        flux=None,
        charge=charge,
        exp_flux=lambda c: _shift_power(shift, c),
        exp_charge=lambda c: sp.diags(np.exp(1j * c * vals), 0, format="csc"),
    )


def dual_charge_mode(cut: int) -> Mode:
    """Compact-charge (phase-slip) mode in the dual charge basis ``|ψ⟩``.

    Mirror image of :func:`charge_mode`: the integer flux ``ψ ∈ [−cut, cut]`` is
    diagonal and the charge ``q`` is periodic, so ``exp(i c q)`` shifts
    ``ψ → ψ + round(c)``.
    """
    vals = np.arange(-cut, cut + 1)
    dim = 2 * cut + 1
    flux = sp.diags(vals, 0, format="csc", dtype=float)
    shift = _charge_shift(dim)
    return Mode(
        dim=dim,
        flux=flux,
        charge=None,
        exp_flux=lambda c: sp.diags(np.exp(1j * c * vals), 0, format="csc"),
        exp_charge=lambda c: _shift_power(shift, c),
    )
