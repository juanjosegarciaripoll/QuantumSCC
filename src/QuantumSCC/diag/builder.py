"""
Sparse Hamiltonian construction and diagonalisation for QuantumSCC circuits.

Consumes a :class:`QuantumSCC.CircuitModel` (or a solved ``Circuit``) and builds
a ``scipy.sparse`` Hamiltonian following the recipe in QuantumSCC's
``docs/model_extraction.md``:

- one Hilbert space per mode, chosen from its sector — charge basis for
  compact-flux (JJ) and compact-charge (QPS) modes, Fock basis for extended
  (oscillator) modes (see :mod:`QuantumSCC.diag.operators`);
- the quadratic part ``½ φᵀ K_flux φ + ½ nᵀ K_charge n`` assembled from
  ``K_flux`` / ``K_charge``;
- the cosines assembled from the coupling vectors, factorised over modes as
  ``exp(i Σ_m c_m φ_m) = ⊗_m exp(i c_m φ_m)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from .operators import Mode, charge_mode, dual_charge_mode, oscillator_mode

type Sparse = Any

__all__ = ["build_hamiltonian", "eigenenergies", "eigenstates"]

_TOL = 1e-12


def _as_model(obj: Any) -> Any:
    """Accept either a CircuitModel (has ``K_flux``) or a Circuit (has ``model``)."""
    if hasattr(obj, "K_flux"):
        return obj
    if hasattr(obj, "model"):
        return obj.model()
    raise TypeError("expected a QuantumSCC Circuit or CircuitModel")


def _modes(model: Any, charge_cut: int, fock_dim: int) -> list[Mode]:
    """Build one :class:`Mode` per circuit mode, dispatched on its sector."""
    modes: list[Mode] = []
    for sector in model.mode_sectors:
        if sector == "extended":
            modes.append(oscillator_mode(fock_dim))
        elif sector == "compact_flux":
            modes.append(charge_mode(charge_cut))
        elif sector == "compact_charge":
            modes.append(dual_charge_mode(charge_cut))
        else:  # pragma: no cover - guards against an upstream sector rename
            raise ValueError(f"unknown mode sector {sector!r}")
    return modes


def _embed(op: Sparse, m: int, dims: list[int]) -> Sparse:
    """Place single-mode operator ``op`` (on mode ``m``) into the full space."""
    out = sp.identity(1, format="csc", dtype=complex)
    for k, dim in enumerate(dims):
        factor = op if k == m else sp.identity(dim, format="csc")
        out = sp.kron(out, factor, format="csc")
    return out


def _quadratic(
    K: npt.NDArray[np.float64],
    ops: list[Sparse | None],
    dims: list[int],
    dim: int,
) -> Sparse:
    """``½ Σ_{a,b} K[a,b] · x_a x_b`` for a coordinate family (flux or charge)."""
    H = sp.csc_matrix((dim, dim), dtype=complex)
    n = len(dims)
    for a in range(n):
        for b in range(n):
            if abs(K[a, b]) < _TOL:
                continue
            if ops[a] is None or ops[b] is None:
                raise ValueError(
                    "quadratic part references a periodic operator — the model "
                    "violates the 'periodic operators only in cosines' invariant; "
                    "cannot reconstruct."
                )
            H = H + 0.5 * K[a, b] * (_embed(ops[a], a, dims) @ _embed(ops[b], b, dims))
    return H


def _cosines(
    amplitudes: npt.NDArray[np.float64],
    vectors: npt.NDArray[np.float64],
    exps: list[Any],
    dims: list[int],
    dim: int,
) -> Sparse:
    """``− Σ_t E_t cos(Σ_m V[t,m] x_m)`` via ``exp(iΣ) = ⊗ exp(i·)``."""
    H = sp.csc_matrix((dim, dim), dtype=complex)
    for t in range(len(amplitudes)):
        P = sp.identity(dim, format="csc", dtype=complex)
        for m in range(len(dims)):
            coeff = vectors[t, m]
            if abs(coeff) >= _TOL:
                P = P @ _embed(exps[m](coeff), m, dims)
        H = H - amplitudes[t] * 0.5 * (P + P.conj().T)
    return H


def build_hamiltonian(model: Any, charge_cut: int = 12, fock_dim: int = 12) -> Sparse:
    """Build the sparse Hamiltonian ``H/ℏ`` (GHz) for ``model``.

    Parameters
    ----------
    model : QuantumSCC.CircuitModel or QuantumSCC.Circuit
        The circuit model to reconstruct. A ``Circuit`` is converted via
        ``circuit.model()``.
    charge_cut : int
        Charge-basis cutoff for compact modes: each uses ``2*charge_cut + 1``
        states ``|n⟩``, ``n ∈ [−charge_cut, charge_cut]``.
    fock_dim : int
        Fock-space dimension for each extended (oscillator) mode.

    Returns
    -------
    scipy.sparse.csc_matrix
        The Hamiltonian ``H/ℏ`` on the tensor-product space.
    """
    model = _as_model(model)
    modes = _modes(model, charge_cut, fock_dim)
    dims = [mode.dim for mode in modes]
    dim = int(np.prod(dims)) if dims else 1

    flux_ops = [mode.flux for mode in modes]
    charge_ops = [mode.charge for mode in modes]

    H = sp.csc_matrix((dim, dim), dtype=complex)
    H = H + _quadratic(np.asarray(model.K_flux, dtype=float), flux_ops, dims, dim)
    H = H + _quadratic(np.asarray(model.K_charge, dtype=float), charge_ops, dims, dim)

    # JJ cosines act on the flux coordinates, QPS cosines on the charge ones.
    H = H + _cosines(
        np.asarray(model.E_J, dtype=float),
        np.asarray(model.v_J, dtype=float),
        [mode.exp_flux for mode in modes],
        dims,
        dim,
    )
    H = H + _cosines(
        np.asarray(model.E_P, dtype=float),
        np.asarray(model.v_P, dtype=float),
        [mode.exp_charge for mode in modes],
        dims,
        dim,
    )
    return H.tocsc()


def _lowest(H: Sparse, k: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """The ``k`` lowest eigenpairs of ``H``, ascending; dense for tiny spaces."""
    if H.shape[0] <= max(k + 1, 2):
        vals, vecs = np.linalg.eigh(H.toarray())
        return vals[:k].real, vecs[:, :k]
    from scipy.sparse.linalg import eigsh

    vals, vecs = eigsh(H, k=k, which="SA")
    order = np.argsort(vals.real)
    return vals[order].real, vecs[:, order]


def eigenenergies(
    model: Any,
    k: int = 6,
    charge_cut: int = 12,
    fock_dim: int = 12,
    relative: bool = False,
) -> npt.NDArray[np.float64]:
    """Return the ``k`` lowest eigenenergies (GHz), sorted ascending.

    With ``relative=True`` the ground-state energy is subtracted, giving
    transition energies from the ground state.
    """
    H = build_hamiltonian(model, charge_cut=charge_cut, fock_dim=fock_dim)
    vals, _ = _lowest(H, k)
    return vals - vals[0] if relative else vals


def eigenstates(
    model: Any,
    k: int = 6,
    charge_cut: int = 12,
    fock_dim: int = 12,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Return the ``k`` lowest ``(eigenenergies, eigenvectors)`` of ``model``.

    Eigenvalues are ascending (GHz); ``eigenvectors[:, i]`` is the state for
    ``eigenenergies[i]`` on the tensor-product space.
    """
    H = build_hamiltonian(model, charge_cut=charge_cut, fock_dim=fock_dim)
    return _lowest(H, k)
