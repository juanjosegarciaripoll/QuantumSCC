"""
model.py — a self-contained, solver-ready description of a quantised circuit.

`CircuitModel` repackages the result of the QuantumSCC pipeline (in the
"phiq" / ready-to-quantise basis) into a single flat data structure from
which any backend can reconstruct the Hamiltonian without knowing the
internal index conventions of the pipeline.

The model represents

    H/ℏ = ½ φᵀ K_flux φ  +  ½ nᵀ K_charge n
          − Σ_j E_J[j] · cos( Σ_m v_J[j, m] · φ_m )
          − Σ_k E_P[k] · cos( Σ_m v_P[k, m] · n_m )

where, for each of the N = n_compact_flux + n_compact_charge + n_extended
modes, φ_m is a flux operator and n_m its conjugate charge operator.

The mode order is fixed:

    [ compact_flux | compact_charge | extended ]

so a mode's index alone determines its sector (and hence which Hilbert
space a solver should use for it). See ``docs/model_extraction.md`` for the
full reconstruction recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .core.elements import Junction, PhaseSlip

if TYPE_CHECKING:
    from .circuit import Circuit

SECTORS = ("compact_flux", "compact_charge", "extended")


@dataclass
class CircuitModel:
    """Solver-ready Hamiltonian data in the phiq (ready-to-quantise) basis.

    All energies are in ``units`` (GHz by default), following the
    conventions ``n = Q/2e`` and ``φ = 2π Φ/Φ₀``.

    Attributes
    ----------
    n_compact_flux : int
        Number of Josephson (compact-flux) modes, pairs ``(φ_c, n_c)``.
        Suggested Hilbert space: charge basis (integer ``n_c``).
    n_compact_charge : int
        Number of quantum-phase-slip (compact-charge) modes, pairs
        ``(ψ_c, q_c)``. Suggested Hilbert space: dual charge basis.
    n_extended : int
        Number of harmonic-oscillator modes, pairs ``(φ_e, n_e)``.
        Suggested Hilbert space: Fock basis. Each extended mode is in
        isotropic normal-mode form, so ``K_flux[m, m] == K_charge[m, m]``
        equals the mode frequency ω.
    K_flux, K_charge : numpy.ndarray
        ``(N, N)`` real symmetric matrices of the quadratic part. Diagonal
        entries are per-mode self-energies; off-diagonal entries are the
        residual cross-sector couplings that survive the phiq reduction.
        Convention: ``H_quad/ℏ = ½ φᵀ K_flux φ + ½ nᵀ K_charge n``.
    E_J, v_J : numpy.ndarray
        Josephson amplitudes ``(n_JJ,)`` and coupling vectors
        ``(n_JJ, N)`` over the *flux* coordinates.
    E_P, v_P : numpy.ndarray
        Phase-slip amplitudes ``(n_QPS,)`` and coupling vectors
        ``(n_QPS, N)`` over the *charge* coordinates.
    units : str
        Energy/frequency unit of all amplitudes and matrices.
    """

    n_compact_flux: int
    n_compact_charge: int
    n_extended: int

    K_flux: np.ndarray
    K_charge: np.ndarray

    E_J: np.ndarray
    v_J: np.ndarray

    E_P: np.ndarray
    v_P: np.ndarray

    units: str = "GHz"

    # ── derived convenience ────────────────────────────────────────────
    @property
    def n_modes(self) -> int:
        """Total number of conjugate-variable pairs ``N``."""
        return self.n_compact_flux + self.n_compact_charge + self.n_extended

    def sector(self, m: int) -> str:
        """Return the sector name of mode ``m`` (``'compact_flux'``,
        ``'compact_charge'`` or ``'extended'``)."""
        if not 0 <= m < self.n_modes:
            raise IndexError(f"mode index {m} out of range [0, {self.n_modes})")
        if m < self.n_compact_flux:
            return "compact_flux"
        if m < self.n_compact_flux + self.n_compact_charge:
            return "compact_charge"
        return "extended"

    @property
    def mode_sectors(self) -> tuple[str, ...]:
        """Sector name of every mode, in index order."""
        return tuple(self.sector(m) for m in range(self.n_modes))

    def mode_label(self, m: int) -> tuple[str, str]:
        """Return the ``(flux_label, charge_label)`` of mode ``m`` matching
        the symbols used by the printout functions, e.g. ``('phi_c1', 'n_c1')``."""
        s = self.sector(m)
        if s == "compact_flux":
            k = m + 1
            return (f"phi_c{k}", f"n_c{k}")
        if s == "compact_charge":
            k = m - self.n_compact_flux + 1
            return (f"psi_c{k}", f"q_c{k}")
        k = m - self.n_compact_flux - self.n_compact_charge + 1
        return (f"phi_e{k}", f"n_e{k}")

    # ── construction ───────────────────────────────────────────────────
    @classmethod
    def from_circuit(cls, circuit: Circuit, tol: float = 1e-12) -> CircuitModel:
        """Build a :class:`CircuitModel` from a solved :class:`~QuantumSCC.Circuit`.

        The data is read from the phiq (ready-to-quantise) basis. Entries with
        magnitude below ``tol`` are zeroed so the quadratic matrices and
        coupling vectors are sparse; pass ``tol=0`` to keep raw values.
        """
        N = circuit.no_independent_variables // 2

        H = np.asarray(circuit.FS_quadratic_hamiltonian_phiq).real
        K_flux = H[:N, :N].copy()
        K_charge = H[N:, N:].copy()
        # Symmetrise to remove tiny numerical asymmetry, then sparsify.
        K_flux = 0.5 * (K_flux + K_flux.T)
        K_charge = 0.5 * (K_charge + K_charge.T)

        # JJ cosines live in the flux coordinates (rows [:N]); QPS cosines in
        # the charge coordinates (rows [N:]). Transpose to (n_terms, N).
        vJ_full = np.asarray(circuit.final_vector_JJ_phiq).real
        vP_full = np.asarray(circuit.final_vector_QPS_phiq).real
        v_J = vJ_full[:N, :].T.copy()
        v_P = vP_full[N:, :].T.copy()

        E_J = np.array(
            [e[2].value() for e in circuit.elements if isinstance(e[2], Junction)],
            dtype=float,
        )
        E_P = np.array(
            [e[2].value() for e in circuit.elements if isinstance(e[2], PhaseSlip)],
            dtype=float,
        )

        if tol:
            for arr in (K_flux, K_charge, v_J, v_P):
                arr[np.abs(arr) < tol] = 0.0

        return cls(
            n_compact_flux=int(circuit.no_final_compact_flux),
            n_compact_charge=int(circuit.no_final_compact_charge),
            n_extended=int(N - circuit.no_final_compact_flux - circuit.no_final_compact_charge),
            K_flux=K_flux,
            K_charge=K_charge,
            E_J=E_J,
            v_J=v_J,
            E_P=E_P,
            v_P=v_P,
        )

    def __repr__(self) -> str:
        return (
            f"CircuitModel(n_modes={self.n_modes}, "
            f"compact_flux={self.n_compact_flux}, "
            f"compact_charge={self.n_compact_charge}, "
            f"extended={self.n_extended}, "
            f"n_JJ={len(self.E_J)}, n_QPS={len(self.E_P)}, units='{self.units}')"
        )
