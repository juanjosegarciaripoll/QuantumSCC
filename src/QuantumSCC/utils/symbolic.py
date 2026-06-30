"""
utils/symbolic.py

Builds the symbolic Hamiltonian from the numerical pipeline results.
K and V are purely topological (no energy dependence), so the quadratic
Hamiltonian is linear in the energy parameters:

    H_quad = sum_i 2*E_L_i * (v_i v_i^T) + sum_j 2*E_C_j * (w_j w_j^T)

where v_i = (V^T K^T)[:, inductor_col] and w_j = (V^T K^T)[:, cap_charge_col]
are numerical vectors extracted from the existing numerical pipeline.

This avoids full sympy matrix multiplication — only scalar * outer product additions.
The Schur complement (Eq. 18 of the paper) is applied symbolically if needed.
"""

import sympy as sp
import numpy as np

from ..core.elements import Capacitor, Inductor, Junction, PhaseSlip


def _to_sym(x, tol=1e-9):
    """Convert a float coefficient to a clean sympy expression."""
    if abs(x) < tol:
        return sp.Integer(0)
    if abs(x - round(x)) < tol:
        return sp.Integer(int(round(x)))
    return sp.nsimplify(x, [sp.sqrt(2), sp.sqrt(3), sp.sqrt(6)],
                        rational=False, tolerance=1e-7)


def build_symbolic_hamiltonian(topo, geom):
    """
    Build the symbolic quadratic Hamiltonian using coefficient extraction.

    Since K and V depend only on circuit topology (not on E_C, E_L, E_J, E_P),
    H_quad = sum_k 2*E_k * col_k * col_k^T  (linear in energy parameters).

    The Schur complement is applied symbolically to eliminate non-dynamical
    variables, matching the numerical pipeline in classical_hamiltonian_function().

    Parameters
    ----------
    topo : Topology
    geom : Geometry

    Returns
    -------
    H_sym : sympy.Matrix  (no_indep × no_indep)
        Symbolic quadratic Hamiltonian (before normal-mode diagonalisation).
    sym_vals : dict {sympy.Symbol → float}
        Numerical values of E_C_k and E_L_k symbols.
    J_syms : list of (sympy.Symbol, float)
        (E_J_k symbol, numerical value) for each Junction.
    P_syms : list of (sympy.Symbol, float)
        (E_P_k symbol, numerical value) for each PhaseSlip.
    """
    K          = topo.K
    V          = geom.V
    no_elements = topo.no_elements
    no_indep   = geom.no_independent_variables

    # VTK[row, col] = (V^T K^T)[row, col]
    # col i            → flux sector of element i
    # col i+no_elements → charge sector of element i
    VTK = V.T @ K.T          # shape: (V.shape[0], 2*no_elements)
    n_full = VTK.shape[0]    # may be > no_indep when gauge variables exist

    # ── Count each element type for symbol naming ──────────────────────────
    n_cap = sum(1 for row in topo.elements if isinstance(row[2], Capacitor))
    n_ind = sum(1 for row in topo.elements if isinstance(row[2], Inductor))
    n_jj  = topo.no_JJ
    n_qps = topo.no_QPS

    # ── Build H_full = Σ_k (E_k/2) · col_k col_k^T  (Adrián convention) ──
    # E_k symbols represent E_C = 4e²/C, E_L = (Φ₀/2π)²/L  (= 2·energy())
    # so H = (E_C/2)·n² + (E_L/2)·φ² matches Adrián's notation.
    H_full   = sp.zeros(n_full, n_full)
    sym_vals = {}
    cap_idx  = 0
    ind_idx  = 0

    for i, elem_row in enumerate(topo.elements):
        elem = elem_row[2]

        if isinstance(elem, Inductor):
            ind_idx += 1
            name = 'E_L' if n_ind == 1 else f'E_L{ind_idx}'
            sym  = sp.Symbol(name, positive=True)
            sym_vals[sym] = 2 * elem.energy()   # E_L_Adrián = 2·E_L_code
            col = sp.Matrix([_to_sym(x) for x in VTK[:, i]])
            H_full += sym * sp.Rational(1, 2) * (col * col.T)

        elif isinstance(elem, Capacitor):
            cap_idx += 1
            name = 'E_C' if n_cap == 1 else f'E_C{cap_idx}'
            sym  = sp.Symbol(name, positive=True)
            sym_vals[sym] = 2 * elem.energy()   # E_C_Adrián = 2·E_C_code
            col = sp.Matrix([_to_sym(x) for x in VTK[:, i + no_elements]])
            H_full += sym * sp.Rational(1, 2) * (col * col.T)

    # ── Schur complement (mirrors classical_hamiltonian_function logic) ────
    if H_full.shape[0] == no_indep:
        H_sym = H_full
    else:
        TEF_11 = H_full[:no_indep, :no_indep]
        TEF_12  = H_full[:no_indep, no_indep:]
        TEF_21  = H_full[no_indep:, :no_indep]
        TEF_22  = H_full[no_indep:, no_indep:]

        subs_num = {sym: float(val) for sym, val in sym_vals.items()}

        def _num(M):
            return np.array(
                [[float(M[i, j].subs(subs_num)) for j in range(M.shape[1])]
                 for i in range(M.shape[0])], dtype=float)

        TEF_22_num = _num(TEF_22)
        max_abs    = np.max(np.abs(TEF_22_num)) if TEF_22_num.size > 0 else 0.0
        threshold  = max(max_abs * 1e-8, 1e-12)

        # Step 1: remove pure-gauge rows (zero rows → rank-0 contribution)
        nz = [i for i in range(TEF_22_num.shape[0])
              if np.any(np.abs(TEF_22_num[i, :]) > threshold)]

        if not nz:
            H_sym = TEF_11
        else:
            T22_red_num = TEF_22_num[np.ix_(nz, nz)]
            rank = np.linalg.matrix_rank(
                T22_red_num,
                tol=max(np.max(np.abs(T22_red_num)) * 1e-8, 1e-12))

            if rank == T22_red_num.shape[0]:
                # Full rank → exact symbolic inverse
                T22_red = TEF_22[nz, :][:, nz]
                T12_red = TEF_12[:, nz]
                T21_red = TEF_21[nz, :]
                H_sym = sp.expand(TEF_11 - T12_red * T22_red.inv() * T21_red)
            else:
                # Rank-deficient → numerical Moore-Penrose pinv + symbolic reconstruction.
                # The Schur correction is linear in energy params for physical circuits,
                # so we reconstruct each entry as Σ_k ratio_k * E_k.
                T22_pinv   = np.linalg.pinv(T22_red_num)
                T12_num    = _num(TEF_12)[:, nz]
                T21_num    = _num(TEF_21)[nz, :]
                schur_num  = T12_num @ T22_pinv @ T21_num

                schur_sym = sp.zeros(no_indep, no_indep)
                for i in range(no_indep):
                    for j in range(no_indep):
                        v = float(schur_num[i, j])
                        if abs(v) < threshold:
                            continue
                        entry_sym = sp.Integer(0)
                        residual  = v
                        for sym, sym_val in sym_vals.items():
                            if abs(sym_val) < 1e-30 or abs(residual) < 1e-9:
                                continue
                            ratio_clean = _to_sym(residual / sym_val, tol=1e-6)
                            if ratio_clean != 0:
                                entry_sym += ratio_clean * sym
                                residual  -= float(ratio_clean) * sym_val
                        schur_sym[i, j] = entry_sym

                H_sym = sp.expand(TEF_11 - schur_sym)

    # ── JJ and QPS symbols ─────────────────────────────────────────────────
    J_syms  = []
    jj_idx  = 0
    for row in topo.elements:
        if isinstance(row[2], Junction):
            jj_idx += 1
            name = 'E_J' if n_jj == 1 else f'E_J{jj_idx}'
            sym  = sp.Symbol(name, positive=True)
            J_syms.append((sym, row[2].value()))

    P_syms  = []
    qps_idx = 0
    for row in topo.elements:
        if isinstance(row[2], PhaseSlip):
            qps_idx += 1
            name = 'E_P' if n_qps == 1 else f'E_P{qps_idx}'
            sym  = sp.Symbol(name, positive=True)
            P_syms.append((sym, row[2].value()))

    return H_sym, sym_vals, J_syms, P_syms
