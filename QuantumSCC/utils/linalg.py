
"""
algebra.py contains the  algebraic functions the program needs to its correct operation
"""

import numpy as np
from fractions import Fraction
from math import gcd
from functools import reduce
from scipy.linalg import null_space

Matrix = np.ndarray


def integer_null_space(M: np.ndarray) -> np.ndarray:
    """
    Compute the integer null space of a matrix with rational entries.

    Uses exact arithmetic (fractions.Fraction) to compute RREF,
    then back-substitutes to produce kernel vectors with integer entries.
    Each column is normalized so that entries are coprime integers.

    Parameters
    ----------
    M : np.ndarray
        Input matrix (m x n) with integer or rational entries.

    Returns
    -------
    K : np.ndarray
        Integer kernel matrix (n x d) where d = n - rank(M).
        M @ K = 0 (exact). Entries are integers (typically 0, ±1).
        Returns shape (n, 0) if kernel is trivial.
    """
    m, n = M.shape
    if m == 0:
        return np.eye(n, dtype=float)

    # Convert to exact Fraction arithmetic
    R = [[Fraction(M[i, j]).limit_denominator(10**12) for j in range(n)] for i in range(m)]

    # Forward elimination with partial pivoting → RREF
    pivot_cols = []
    pivot_row = 0
    for col in range(n):
        # Find pivot in this column
        found = -1
        for row in range(pivot_row, m):
            if R[row][col] != 0:
                found = row
                break
        if found == -1:
            continue

        # Swap rows
        R[pivot_row], R[found] = R[found], R[pivot_row]

        # Scale pivot row
        scale = R[pivot_row][col]
        for j in range(n):
            R[pivot_row][j] /= scale

        # Eliminate all other rows
        for row in range(m):
            if row == pivot_row:
                continue
            factor = R[row][col]
            if factor != 0:
                for j in range(n):
                    R[row][j] -= factor * R[pivot_row][j]

        pivot_cols.append(col)
        pivot_row += 1

    rank = len(pivot_cols)
    free_cols = [j for j in range(n) if j not in pivot_cols]

    if len(free_cols) == 0:
        return np.zeros((n, 0), dtype=float)

    # Build kernel vectors: for each free column, set it to 1
    # and read off pivot values from RREF
    K_cols = []
    for fc in free_cols:
        vec = [Fraction(0)] * n
        vec[fc] = Fraction(1)
        for i, pc in enumerate(pivot_cols):
            vec[pc] = -R[i][fc]
        K_cols.append(vec)

    # Convert to integer vectors (multiply by LCM of denominators)
    K = np.zeros((n, len(K_cols)), dtype=float)
    for j, vec in enumerate(K_cols):
        denoms = [abs(v.denominator) for v in vec if v != 0]
        if denoms:
            lcm_val = reduce(lambda a, b: a * b // gcd(a, b), denoms)
        else:
            lcm_val = 1
        int_vec = [int(v * lcm_val) for v in vec]
        # Normalize by GCD
        g = reduce(gcd, [abs(x) for x in int_vec if x != 0], 0)
        if g > 0:
            int_vec = [x // g for x in int_vec]
        K[:, j] = int_vec

    return K

def GaussJordan(M: Matrix):
    """
    Transform the matrix M in an upper triangular matrix using the Gauss-Jordan algorithm.

    Parameters
    ----------
        M: Matrix
            Matrix to which the algorithm is applied.

    Returns
    ----------
        M: Matrix
            Upper triangular form of the input matrix once the algorithm has been applied.
        order: np.array 
            Variable order of the new upper diagonal matrix.
    """
    
    nrows, ncolumns = M.shape
    M = M.copy()
    order = np.arange(ncolumns)
    n_pivots = min(nrows, ncolumns)
    for i in range(n_pivots):
        k = np.argmax(np.abs(M[i, i:]))
        if k != 0:
            Maux = M.copy()
            M[:, i], M[:, i + k] = Maux[:, i + k], Maux[:, i]
            order[i], order[i + k] = order[i + k], order[i]
        if np.abs(M[i, i]) < 1e-15:
            continue
        for j in range(i + 1, nrows):
            M[j, :] -= M[i, :] * M[j, i] / M[i, i]

    return M, order


def reverseGaussJordan(M: Matrix):
    """
    Transform an upper triangular matrix, M, into a diagonal matrix using the Gauss-Jordan algorithm.

    Parameters
    ----------
        M: Matrix
            Upper triangular matrix to which the algorithm is applied.

    Returns
    ----------
        M: Matrix
            Diagonal form of the input matrix once the algorithm has been applied.
    """

    M = np.diag(1.0 / np.diag(M)) @ M

    for i, row in reversed(list(enumerate(M))):
        for j in range(i):
            M[j, :] -= M[j, i] * row

    return M


def remove_zero_rows(M: Matrix, tol: float=1e-16):
    """
    Removes all-zero rows from a matrix M.

    Parameters
    ----------
        M: Matrix
            Matrix to which the algorithm is applied.
        tol: float
            Tolerance below which a element is considered zero. By default, it is 1e-16.

    Returns
    ----------
        M: Matrix 
            Input matrix with no zero rows.
    """

    row_norm_1 = np.sum(np.abs(M), -1)
    M = M[(row_norm_1 > tol), :]
    return M


def pseudo_inv(M: Matrix, tol: float=1e-15):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and considering a total
    tolerance below which each singular value is considered zero.

    Parameters
    ----------
        M: Matrix
            Input matrix 
        tol: float
            Tolerance below which the element is considered zero. By default, it is 1e-15.

    Returns
    ----------
        pseudo_inv: Matrix
            Moore-Penrose pseudo-inverse matrix of input matrix.
    """
    # SVD decomposition
    U, S, Vt = np.linalg.svd(M)
    
    # Invert the singular values taking into account the tolerance
    S_inv = np.zeros((Vt.shape[0], U.shape[1]))  # Preallocate S_inv matrix with correct dimensions
    for i in range(len(S)):
        if np.abs(S[i]) > tol:  # Invert only if the singular value is bigger than the tolerance
            S_inv[i, i] = 1 / S[i]
    
    # Get the pseudo-inverse
    pseudo_inv = Vt.T @ S_inv @ U.T
    return pseudo_inv


def proportional_rows(M: Matrix , tol: float=1e-14):
    """
    It returns the indexes of the proportional rows, separated by groups, of the input matrix M.

    Parameters
    ----------
        M: Matrix
            Input matrix.
        tol: float
            Tolerance below which the element is considered zero. By default, it is 1e-14.
    Returns
    ----------
        proportional_rows_list: list
            List of list made by the indexes of the proportional rows, separated by groups, of the input matrix M
    """

    no_rows = M.shape[0]
    rows_visited = set()
    proportional_rows_list = []

    for i in range(no_rows):
        if i in rows_visited:
            continue
            
        group = [i]
        rows_visited.add(i)

        for j in range (i + 1, no_rows):
            
            if j in rows_visited:
                continue

            # Check if rows i and j are proportional
            ratio = None
            is_proportional = True
            for x, y in zip(M[i, :], M[j, :]):
                
                if np.abs(x) < tol and np.abs(y) < tol:
                    continue
                elif np.abs(x) < tol or np.abs(y) < tol:
                    is_proportional = False
                    break
                elif ratio is None:
                    ratio = x / y
                elif not np.isclose(ratio, x / y, atol=tol):
                    is_proportional = False
                    break

                
            if is_proportional == True:
                group.append(j)
                rows_visited.add(j)

        # Only add the group to the list if it contains more than one row
        if len(group) > 1:
            proportional_rows_list.append(group)
    
    return proportional_rows_list


def Gauge_variable_symplification(M: Matrix, row_index: int, column_index: int, tol: float=1e-14):
    """
    Performs column operations to make all elements in the specified row (row_index),
    except the element [row_index, column_index], equal to 0.

    Parameters
    ----------
        M: Matrix
            Input matrix.
        row_index: int
            Index of the row we are using to implement the algorithm.
        column_index: int
            Index of the column we are using to implement the algorithm.
        tol: float
            Tolerance below which the element is considered zero. By default, it is 1e-14.
    Returns
    ----------
        M: Matrix
            Modified matrix with the specified row zeroed out.
    """
    M = M.astype(float)  # Ensure floating-point precision for operations
    no_columns = M.shape[1] 

    # Ensure the pivot element [row_index, column_index] is not zero
    if np.abs(M[row_index, column_index]) < tol:
        raise ValueError(f"The pivot element of Kloop at [{row_index}, {column_index}] is zero, cannot proceed with the Gauge variables simplification.")

    # Normalize the column of the pivot to make M[row_index, column_index] = 1
    M[:, column_index] = M[:, column_index] / M[row_index, column_index]

    # Eliminate all other elements in the row
    for i in range(no_columns):
        if i != column_index:  # Skip the pivot column
            factor = M[row_index, i]
            M[:, i] -= factor * M[:, column_index]

    return M

    
def omega_symplectic_transformation(
    Omega: Matrix,
    no_compact_flux_variables: int,
    no_flux_variables: int,
    no_compact_charge_variables: int = 0,
    tol: float = 1e-14,
) -> tuple:
    """
    Transform an antisymmetric matrix Omega to the symplectic matrix J such that
    J = V.T @ Omega @ V, using the systematic Darboux reduction from
    QPS-JJ-reduction.pdf (PRX 2025 procedure).

    Variable ordering of Omega: [φ_S | φ_R | Q_S | Q_R]
      φ_S = compact flux   (nCF)     Q_S = compact charge  (nCC)
      φ_R = extended flux   (nEF)     Q_R = extended charge (nEC)

    Block structure (Eq. 42 two-topology):
                 φ_S    φ_R    Q_S    Q_R
        φ_S  [   0      0      0      A   ]
        φ_R  [   0      0    -B^T   -D^T  ]
        Q_S  [   0      B      0      0   ]
        Q_R  [  -A^T    D      0      *   ]

    where (QPS-JJ-reduction.pdf):
      A = ω[φ_S, Q_R]  (nCF × nEC): compact flux ↔ extended charge
      B = ω[Q_S, φ_R]  (nCC × nEF): compact charge ↔ extended flux
      D = ω[Q_R, φ_R]  (nEC × nEF): extended charge ↔ extended flux

    Pairing order (PDF systematic reduction):
      Phase 1: φ_S ↔ Q_R via A  (compact flux picks extended charge)
      Phase 2: φ_R ↔ Q_S via B  (extended flux picks compact charge)
      Phase 3: φ_R ↔ Q_R via D  (remaining extended flux picks remaining Q_R)

    Parameters
    ----------
        Omega: array_like
            Antisymmetric matrix to which the algorithm is applied.
        no_compact_flux_variables: int
            Number of compact flux variables (JJ/Cap sector).
        no_flux_variables: int
            Total number of flux variables.
        no_compact_charge_variables: int
            Number of compact charge variables (QPS sector). Default 0 (backwards compatible).
        tol: float
            Tolerance below which an element is considered zero. Default 1e-14.

    Returns
    ----------
        J: Matrix
            Symplectic matrix from the transformation V.T @ Omega @ V.
        V: Matrix
            Basis change matrix that transforms Omega into J.
        no_compact_flux_variables: int
            Number of independent compact flux variables after gauge deletion.
        no_compact_charge_variables: int
            Number of independent compact charge variables after gauge deletion.
    """

    # Define the number of extended flux variables and charge variables
    no_extended_flux_variables = no_flux_variables - no_compact_flux_variables
    no_charge_variables = Omega.shape[0] - no_flux_variables

    # Verify that the input matrix Omega is antisymmetric
    assert np.allclose(Omega.T, - Omega), "Input Omega matrix must be antisymmetric"

    # Delete gauge variables (all-zero rows and columns in Omega)
    Omega_new = Omega.copy()
    delete_index_list = []
    for i in range(Omega.shape[0]):
        if np.all(np.abs(Omega[i, :]) < tol):
            delete_index_list.append(i)

    Omega_new = np.delete(Omega_new, delete_index_list, axis=0)
    Omega_new = np.delete(Omega_new, delete_index_list, axis=1)

    # Update variable counts after gauge deletion.
    # Original ordering boundaries: [0, nCF) compact flux, [nCF, nF) extended flux,
    # [nF, nF+nCC) compact charge, [nF+nCC, end) extended charge.
    aux_nCF = no_compact_flux_variables
    aux_nF  = no_flux_variables
    aux_nCC = no_flux_variables + no_compact_charge_variables

    for delete_index in delete_index_list:
        if delete_index < aux_nCF:
            no_compact_flux_variables  -= 1
            no_flux_variables          -= 1
        elif aux_nCF <= delete_index < aux_nF:
            no_extended_flux_variables -= 1
            no_flux_variables          -= 1
        elif aux_nF <= delete_index < aux_nCC:
            no_compact_charge_variables -= 1
            no_charge_variables         -= 1
        else:
            no_charge_variables -= 1

    no_extended_flux_variables = no_flux_variables - no_compact_flux_variables

    nCF = no_compact_flux_variables
    nF  = no_flux_variables
    nQ  = no_charge_variables
    nEF = no_extended_flux_variables
    nCC = no_compact_charge_variables

    # ── Hidden gauge detection via T_QR rotation (QPS-JJ-reduction.pdf §2-4)
    #
    # When nF > nQ, there are flux gauge directions that are non-coordinate-
    # aligned linear combinations (e.g., φ_CF - φ_EF), invisible to the
    # all-zero-row scan above.
    #
    # Adrian (QPS-JJ-reduction.pdf p.3) defines the rotation:
    #
    #   T_QR = ( A          )     where A = Ω[φ_S, Q_R]
    #          ( basis(A⊥)  )
    #
    # This splits Q_R into: Q̃_RS (paired with φ_S) and Q̃_RR (remainder).
    # Analogously T_ΦR = [B; basis(B⊥)] for the flux sector.
    # After applying T_QR and T_ΦR, gauge directions become coordinate-
    # aligned (last rows of the rotated Ω are zero) and can be deleted.
    # The remaining block D' = T_QR^T · D · T_ΦR^{-1} is then reduced
    # by the standard Darboux algorithm.
    #
    # Implementation: we use SVD of Ω_FC as a numerically stable way to
    # compute "basis(A⊥)". The SVD Ω_FC = U·Σ·V^T gives:
    #   U[:, :rank]  = range(Ω_FC)   → dynamical flux directions
    #   U[:, rank:]  = ker(Ω_FC^T)   → flux gauge directions ("basis(A⊥)")
    # So U^T is exactly T_QR applied to the flux sector.
    # The SVD is not prescribed by the paper — it is our numerical choice
    # to implement the abstract "basis of A⊥" construction.
    svd_rotation = None
    svd_delete_hidden = []
    nF_pre_svd = nF
    nQ_pre_svd = nQ

    if nF > nQ and nQ > 0:
        Omega_FC = Omega_new[:nF, nF:]
        rank_FC = np.linalg.matrix_rank(Omega_FC, tol=tol)
        n_flux_gauge = nF - rank_FC

        if n_flux_gauge > 0:
            U, S, Vt = np.linalg.svd(Omega_FC, full_matrices=True)

            # Reorder dynamical columns: purely-compact first, then mixed/extended.
            # A dynamical direction j is compact iff U[nCF:, j] ≈ 0
            # (lies entirely within the original compact flux subspace).
            compact_dyn = [j for j in range(rank_FC)
                           if np.all(np.abs(U[nCF:, j]) < tol)]
            extended_dyn = [j for j in range(rank_FC)
                            if not np.all(np.abs(U[nCF:, j]) < tol)]
            gauge_cols = list(range(rank_FC, nF))
            perm_cols = compact_dyn + extended_dyn + gauge_cols
            U = U[:, perm_cols]

            # Build T_QR rotation: R = block_diag(U^T, I_charge)
            # U^T implements T_QR = [A; basis(A⊥)] for the flux sector.
            R = np.eye(Omega_new.shape[0])
            R[:nF, :nF] = U.T

            Omega_new = R @ Omega_new @ R.T

            # Last n_flux_gauge flux rows/cols are now zero — delete them
            svd_delete_hidden = list(range(nF - n_flux_gauge, nF))
            Omega_new = np.delete(Omega_new, svd_delete_hidden, axis=0)
            Omega_new = np.delete(Omega_new, svd_delete_hidden, axis=1)

            svd_rotation = R

            # Update variable counts
            new_nCF = len(compact_dyn)
            nCF = new_nCF
            nF = rank_FC
            nEF = nF - nCF
            no_compact_flux_variables = nCF
            no_flux_variables = nF
            no_extended_flux_variables = nEF

    # ── Deterministic charge pairing (QPS-JJ-reduction.pdf) ──────────────
    # With correctly constructed K (Eq. 42 integer kernel + [compact|extended]
    # ordering), the block structure of Omega = K^T · omega_2B · K gives:
    #
    #   ω[φ_S, Q_R] = A  (non-zero)     ω[φ_S, Q_S] = 0  (structural zero)
    #   ω[Q_S, φ_R] = B  (non-zero)     ω[φ_S, φ_R] = 0  (structural zero)
    #   ω[Q_R, φ_R] = D  (non-zero)     ω[Q_S, Q_R] = 0  (structural zero)
    #
    # The pairing is determined by the block structure alone — no search needed.
    # charge_perm[i] = j means flux i pairs with charge j (0-indexed in charge block).
    # Charge indices: Q_S at 0..nCC-1, Q_R at nCC..nQ-1.
    charge_perm = []
    available_QR = list(range(nCC, nQ))   # Q_R indices
    available_QS = list(range(nCC))       # Q_S indices

    # Phase 1: φ_S[i] → Q_R[i]  (via A block)
    for i in range(nCF):
        charge_perm.append(available_QR.pop(0))

    # Phase 2: φ_R[j] → Q_S[j]  (via B block)
    # Phase 3: remaining φ_R → Q_R  (via D block)
    for j in range(nEF):
        if available_QS:
            charge_perm.append(available_QS.pop(0))
        elif available_QR:
            charge_perm.append(available_QR.pop(0))

    charge_perm += available_QS + available_QR  # leftover charges (nQ > nF)

    # Permute variables to working order: (compact flux, charges[charge_perm], extended flux).
    # charge_perm reorders the charge block so that the conjugate of each flux variable
    # is placed adjacent to it. For standard circuits charge_perm is the identity.
    charge_perm_col_idx = [nF + charge_perm[j] for j in range(nQ)]
    Omega_perm = np.hstack((
        Omega_new[:, :nCF],
        Omega_new[:, charge_perm_col_idx],
        Omega_new[:, nCF:nF]
    ))
    charge_perm_row_idx = [nF + charge_perm[j] for j in range(nQ)]
    Omega_perm = np.vstack((
        Omega_perm[:nCF, :],
        Omega_perm[charge_perm_row_idx, :],
        Omega_perm[nCF:nF, :]
    ))

    # Validate: compact flux rows must have full rank in the charge columns
    Omega_compact_flux = Omega_perm[:nCF, nCF : nCF + nQ - nEF]
    if len(Omega_compact_flux) > 0:
        if np.linalg.matrix_rank(Omega_compact_flux, tol=tol) < Omega_compact_flux.shape[0]:
            raise ValueError(
                'There are linear dependencies between the rows of Omega. '
                'The program is not yet ready to solve this circuit.'
            )

    # Validate: extended flux rows must have full rank in the charge columns
    Omega_extended_flux = Omega_perm[nCF + nQ:, nCF : nCF + nQ]
    if len(Omega_extended_flux) > 0:
        if np.linalg.matrix_rank(Omega_extended_flux, tol=tol) < Omega_extended_flux.shape[0]:
            raise ValueError(
                'There are linear dependencies between the rows of Omega. '
                'The program is not yet ready to solve this circuit.'
            )

    # Build inv_V in the permuted layout (compact flux, all charges, extended flux).
    n_perm = Omega_perm.shape[0]
    inv_V  = np.zeros((n_perm, n_perm))

    # Block 1: compact flux variables — set unit vector and read conjugate from Omega row
    for i in range(nCF):
        inv_V[i, i] = 1 # canonical position q
        inv_V[i + nCF, :] = Omega_perm[i, :] # canonical momentum p

    # Block 2: extended flux variables — read their Omega rows into the charge block
    for i in range(nEF):
        inv_V[i + 2*nCF, :] = Omega_perm[n_perm - nEF + i, :]

    # Block 3: null-space completion for extra charge variables (nQ > nF case).
    # The null space must be orthogonal to ALL constraint rows in the charge block:
    # - CF conjugate rows: inv_V[nCF : 2*nCF, ...]  (Block 1 output)
    # - EF Omega rows:     inv_V[2*nCF : 2*nCF+nEF, ...]  (Block 2 output)
    # Using only CF conjugate rows (the old nCF>0 branch) misses the EF constraints,
    # causing scipy to pick EC directions over CC when both nCF>0 and nEF>0.
    # The unified formula inv_V[nCF : 2*nCF+nEF, ...] covers both cases:
    #   nCF=0 → slice [0 : nEF] = Block 2 rows only  ✓
    #   nEF=0 → slice [nCF : 2*nCF] = Block 1 rows only  ✓
    #   both  → slice [nCF : 2*nCF+nEF] = Block 1 + Block 2  ✓
    if nQ > nF:
        ns = null_space(inv_V[nCF : 2*nCF + nEF, nCF : nCF + nQ])
        for i in range(nQ - nF):
            inv_V[2*nCF + nEF + i, nCF : nCF + nQ] = ns.T[i, :]

    # Block 4: extended flux identity diagonal
    for i in range(nEF):
        inv_V[nCF + nQ + i, nCF + nQ + i] = 1

    # Unpermute inv_V back to the original variable order.
    # Step 1: undo the (compact flux | charges | extended flux) → (compact flux | extended flux | charges) swap.
    inv_V = np.hstack((
        inv_V[:, :nCF],
        inv_V[:, nCF + nQ:],
        inv_V[:, nCF : nQ + nCF]
    ))
    inv_V = np.vstack((
        inv_V[:nCF, :],
        inv_V[nCF + nQ:, :],
        inv_V[nCF : nQ + nCF, :]
    ))
    # Step 2: undo charge_perm reordering in the charge column block [nF..nF+nQ).
    # After step 1, column nF+i of inv_V holds the coefficient for charge charge_perm[i]
    # (in original charge-index space).  To restore the original Omega_new column order
    # (charge j at column nF+j), apply a column-only permutation with inv_charge_perm.
    # Row order is NOT changed: rows represent canonical variables and their ordering
    # is fixed by the Darboux construction to match J_standard (flux i pairs with row nF+i).
    if nQ > 0:
        inv_charge_perm = list(np.argsort(charge_perm))
        col_idx = list(range(nF)) + [nF + inv_charge_perm[k] for k in range(nQ)]
        inv_V = inv_V[:, col_idx]

    # ── Restore T_QR-rotated gauge variables ────────────────────────────
    # inv_V is currently in the T_QR-reduced coordinate space.  We must:
    #   (a) Re-insert rows/cols for the deleted gauge directions
    #   (b) Compose with R^T to undo the T_QR rotation (R = block_diag(U^T, I))
    # After this, inv_V is back in the zero-row-deleted coordinate space.
    if svd_rotation is not None:
        n_svd_dyn = inv_V.shape[0]
        n_svd_gauges = len(svd_delete_hidden)

        # (a) Insert gauge rows (at the bottom) and columns (at deleted positions)
        inv_V = np.vstack((inv_V, np.zeros((n_svd_gauges, inv_V.shape[1]))))
        for i, idx in enumerate(svd_delete_hidden):
            inv_V = np.hstack((
                inv_V[:, :idx],
                np.zeros((inv_V.shape[0], 1)),
                inv_V[:, idx:]
            ))
            inv_V[n_svd_dyn + i, idx] = 1

        # (b) Compose with T_QR rotation: inv_V was in rotated coords,
        # multiply on the right by R to transform columns back to unrotated.
        # Derivation: V_total = R^T @ V_d ⟹ inv_V_total = inv_V_d @ R
        inv_V = inv_V @ svd_rotation

    # ── Restore zero-row gauge variable rows/cols ────────────────────────
    no_gauge_variables     = len(delete_index_list)
    no_non_gauge_variables = inv_V.shape[0]

    if no_gauge_variables > 0:
        inv_V = np.vstack((inv_V, np.zeros((no_gauge_variables, inv_V.shape[1]))))
        for i, delete_index in enumerate(delete_index_list):
            inv_V = np.hstack((inv_V[:, :delete_index], np.zeros((inv_V.shape[0], 1)), inv_V[:, delete_index:]))
            inv_V[i + no_non_gauge_variables, delete_index] = 1

    V = np.linalg.inv(inv_V)

    J = np.zeros((Omega.shape[0], Omega.shape[1]))
    J[:no_flux_variables, no_flux_variables:2*no_flux_variables] = np.eye(no_flux_variables)
    J[no_flux_variables:2*no_flux_variables, :no_flux_variables] = -np.eye(no_flux_variables)

    assert np.allclose(J, V.T @ Omega @ V), \
        'Something goes wrong. Output matrix V must satisfy J = V.T @ Omega @ V, with J the symplectic matrix'

    return J, V, no_compact_flux_variables, no_compact_charge_variables


def symplectic_transformation(M: Matrix, no_flux_variables: int, tol: float = 1e-14) -> tuple[Matrix, Matrix]:
    """
    Transform a square matrix M = JH (with H a positive semidefinite matrix and J the Symplectic matrix) to eigval*J = [[0,eigval*1],[-eigval*1,0]]
    such that eigval*J = inv(T) @ M @ T.
    Parameters
    ----------
        M: Matrix
            Square matrix to which the algorithm is applied.
        no_flux_variables: int
            Parameter to indicate how many flux variables we have.
        tol: float
            Tolerance below which the element is considered zero. By default, it is 1e-14.
    
    Returns
    ----------
        M_out: Matrix 
            Output matrix. If Omega = False: M_out = eigval*J. If Omega = True: M_out = J
        T: Matrix
            Basis change matrix that transforms the input matrix M into eigval*J (T, Omega = False) or J (V, Omega = True).
    """

    # Verify that the input matrix is square, with an even dimenstion
    assert M.shape[0] == M.shape[1], "The input matrix must be square"

    assert M.shape[0]%2 == 0, "For the case Omega == False, the input matrix must be even"
    

    # Obtain the eigenvalues and eigenvectors of the input matrix and sort them 
    M_eigval, M_eigvec = np.linalg.eig(M)

    index = np.argsort(M_eigval.imag)
    M_eigval = M_eigval[index]
    M_eigvec = M_eigvec[:, index]

    # Verify the input matrix does not have degenerate eigenvalues with geometric multiplicity < algebraic multiplicity
    assert np.linalg.matrix_rank(M_eigvec, tol) == M.shape[0], "There are degenerate eigenvalues with geometric \
        multiplicity < algebraic multiplicity -> I fail my assumption and the program is not ready."

    # Organize the eigenvalues with their eigenvectors in two groups: zero and pure imaginary eigenvalues
    zero_eigval, zero_eigvec = np.empty(0), np.empty((M.shape[1], 0))
    imag_eigval, imag_eigvec = np.empty(0), np.empty((M.shape[1], 0))

    for i, eigval in enumerate(M_eigval):

        if np.allclose(eigval.real, 0) and np.allclose(eigval.imag, 0):
            zero_eigval = np.hstack((zero_eigval, 0)) 
            zero_eigvec = np.hstack((zero_eigvec, M_eigvec[:,i].reshape(-1,1)))

        elif np.allclose(eigval.real, 0) and eigval.imag > 0:
            imag_eigval = np.hstack((imag_eigval, 1j * eigval.imag)) # Positive purely imaginary eigenvalue
            imag_eigvec = np.hstack((imag_eigvec, M_eigvec[:,i].reshape(-1,1)))

    # Verify the input matrix has the correct eigenvalues
    assert 2 * len(imag_eigval) + len(zero_eigval) == len(M_eigval), \
        "The input matrix must have only zero or pure imaginary eigenvalues by conjugate pairs"
    
    # Define the physical symplectic matrix J = [[0, I_nf], [-I_nf, 0]].
    # Using J_phys (not J_code) ensures correct normalization for circuits
    # with zero-energy modes where oscillator eigenvectors span both blocks.
    n = M.shape[0]
    nf = no_flux_variables
    J = np.zeros((n, n))
    J[:nf, nf:2*nf] = np.eye(nf)
    J[nf:2*nf, :nf] = -np.eye(nf)
    
    # Eigenvectors normalization under the symplectic inner product
    normal_imag_eigvec = np.empty((M.shape[0], 0))

    for i, eigval in enumerate(imag_eigval):

        # Repeated eigenvalues
        if i > 0 and np.allclose(imag_eigval[i-1], imag_eigval[i]):
            j += 1 
            summary = 0
            for m in range(1,j+1):
                Phi_star = np.conj(normal_imag_eigvec[:,i-m].T @ J @ np.conj(imag_eigvec[:,i]))
                summary += Phi_star * normal_imag_eigvec[:,i-m].reshape(-1,1) 

            eigvec = (imag_eigvec[:,i].reshape(-1,1) - sigma * summary)
            norm = np.abs(np.sqrt(eigvec.T @ J @ np.conj(eigvec)))
            normal_imag_eigvec = np.hstack((normal_imag_eigvec, eigvec/norm)) 
            continue
        j = 0

        # First eigenvalues
        alpha = imag_eigvec[:,i].T @ J @ np.conj(imag_eigvec[:,i])
        sigma = 1j * np.sign(alpha/1j)
        Phi = np.sqrt(sigma * alpha)
        normal_imag_eigvec = np.hstack((normal_imag_eigvec, (imag_eigvec[:,i].reshape(-1,1))/Phi)) 

        # Verify the orthonormalization of the term i
        assert np.allclose(normal_imag_eigvec[:,i].T @ J @ np.conj(normal_imag_eigvec[:,i]), 1j, rtol = tol) \
            or np.allclose(normal_imag_eigvec[:,i].T @ J @ np.conj(normal_imag_eigvec[:,i]), -1j, rtol = tol), \
            "There is an error in the orthonormalization of an eigenvector from a purely imaginary eigenvalue"
        
    # Add an aditional phase to the imaginary eigenvectors, if it is necessary, to to obtain a block diagonal V matrix if it is possible
    for i in range(len(imag_eigval)):
        if np.allclose(sum(normal_imag_eigvec[:no_flux_variables,i]).real, 0):
            normal_imag_eigvec[:,i] = 1j * normal_imag_eigvec[:,i]

    # Construct the basis change matrix T that brings M to eigval*J
    T_plus = np.empty((n, 0))
    T_minus = np.empty((n, 0))

    for i in range(len(imag_eigval)):
        T_plus = np.hstack((T_plus, np.sqrt(2) * (normal_imag_eigvec[:,i].real).reshape(-1,1)))
        T_minus = np.hstack((T_minus, np.sqrt(2) * (normal_imag_eigvec[:,i].imag).reshape(-1,1)))

    # Handle zero eigenvectors: classify as flux-like or charge-like
    # and form symplectic pairs under J_phys.
    n_zero = len(zero_eigval)
    if n_zero > 0:
        # Recover H from M = JH: H = J^{-1} M = -J M (since J^{-1} = -J)
        H_recovered = (-J @ M).real
        H_recovered = 0.5 * (H_recovered + H_recovered.T)

        H_ff = H_recovered[:nf, :nf]
        H_cc = H_recovered[nf:, nf:]

        # Find null spaces of flux and charge blocks (symmetric PSD)
        eigval_ff, eigvec_ff = np.linalg.eigh(H_ff)
        eigval_cc, eigvec_cc = np.linalg.eigh(H_cc)

        null_ff = eigvec_ff[:, np.abs(eigval_ff) < tol]
        null_cc = eigvec_cc[:, np.abs(eigval_cc) < tol]

        assert null_ff.shape[1] == null_cc.shape[1], \
            "Unbalanced zero modes: flux and charge null spaces have different dimensions"

        # Embed in full space: flux vectors in [v_f, 0], charge vectors in [0, v_c]
        zero_flux_vecs = np.vstack([null_ff, np.zeros((nf, null_ff.shape[1]))])
        zero_charge_vecs = np.vstack([np.zeros((nf, null_cc.shape[1])), null_cc])

        # Normalize symplectic pairing: v_f^T J v_c = 1 for each pair
        for k in range(null_ff.shape[1]):
            beta = zero_flux_vecs[:, k] @ J @ zero_charge_vecs[:, k]
            zero_charge_vecs[:, k] /= beta

        # Assemble T = [osc_flux | zero_flux | osc_charge | zero_charge]
        T = np.hstack((T_plus, zero_flux_vecs, T_minus, zero_charge_vecs))
    else:
        T = np.hstack((T_plus, T_minus))

    # Verify that the matrix T satisies the conditions it must satisfy
    assert T.shape[0] == M.shape[0], "There is an error in the construction of the normal form transfromation matrix T. \
        It must have the same dimension as the input matrix"
    assert np.allclose(J, T.T @ J @ T, rtol = tol), "There is an error in the construction of the normal form transfromation matrix T. \
        It must be symplectic, T.T @ J @ T = J"
    assert np.allclose(T.imag, 0, rtol = tol), "There is an error in the construction of the normal form transfromation matrix T. It must be real"

    # Obtain and return the output matrix 
    M_out = np.linalg.pinv(T) @ M @ T

    return M_out, T