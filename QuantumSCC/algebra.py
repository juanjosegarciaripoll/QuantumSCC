
"""
algebra.py contains the  algebraic functions the program needs to its correct operation
"""

import numpy as np
from scipy.linalg import null_space

Matrix = np.ndarray

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
    assert nrows <= ncolumns, "Kirchhoff matrix dimensions are incorrect."
    M = M.copy()
    order = np.arange(ncolumns)
    for i in range(nrows):
        k = np.argmax(np.abs(M[i, i:]))
        if k != 0:
            Maux = M.copy()
            M[:, i], M[:, i + k] = Maux[:, i + k], Maux[:, i]
            order[i], order[i + k] = order[i + k], order[i]
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

    if False:
        factor = 1 / np.diag(M)
        M = factor[:, np.newaxis] * M
    else:
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


def GS_algorithm(M: Matrix, normal: bool=True, delete_zeros: bool=True ,tol: float=1e-14):
    """
    Apply the Gram-Schmidt (GS) algorithm to the columns of the matrix M.

    Parameters
    ----------
        M: Matrix
            Matrix to which columns the GS algorithm is applied
        normal: bool
            Parameter to indicate if the algorithm normalize the resulting orthogonal vectors or not. 
        delete_zeros: boll
            Parameter to indicate if the algorithm deletes the zero columns or not. By default it is True.
        tol: float
            Tolerance below which a number is considered 0. By default it is 1e-14

    Returns
    ----------
        M_out: Matrix 
            Resulting matrix with the columns being orthognormal (normal=True) or just orthogonal (normal=False)
    """

    # Ensure the first vector is not zero
    if np.all(np.abs(M[:,0]) < tol):
        raise ValueError('The first vector of input matrix M must be different from zero')

    # Preallocate M_out
    M_out = np.zeros((M.shape[0], M.shape[1]))

    # Define and normalize if normal = True the first vectors
    M_out[:,0] = M[:,0]
    if normal == True:
        M_out[:,0] = M_out[:,0]/np.sqrt(M_out[:,0].T @ M_out[:,0])

    # Orthogonalize (normal=false) or orthonormalize (normal=True) the other vectors
    for i in range(1, M.shape[1]):
        sum = np.zeros([M.shape[0], 1])

        for j in range(i):
            if np.any(np.abs(M_out[:,j]) > tol):
                sum[:,0] = sum[:,0] + ((M[:,i].T @ M_out[:,j])/(M_out[:,j].T @ M_out[:,j])) * M_out[:,j] 

        M_out[:,i] = M[:,i] - sum[:,0]

        if normal == True:
            if np.any(np.abs(M_out[:,i]) > tol):
                M_out[:,i] = M_out[:,i]/np.sqrt(M_out[:,i].T @ M_out[:,i])

    # Delete zero columns if delete_zeros = True
    if delete_zeros == True:
        zero_list = []
        for i in range(M_out.shape[1]):
            if not np.any(np.abs(M_out[:, i]) > tol): 
                zero_list.append(i)
                
        M_out = np.delete(M_out, zero_list, axis=1) 

    return M_out
 

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


def nonzero_indexes(M: Matrix, tol: float=1e-14):
    """
    It returns the row indexes of the non-zero elements in the matrix M, with a tolerance.
    Parameters
    ----------
        M: Matrix
            Matrix from which we want the non-zero row indexes.
        tol: float
            Tolerance below which the element is considered zero. By default, it is 1e-14.
    Returns
    ----------
        indexes: list
            List of row indexes of the non-zero elements in the input matrix M.
    """
    indexes = []
    for j in range(M.shape[1]):
        for i in range(M.shape[0]):
            if abs(M[i,j]) > tol:
                indexes.append(i)

    indexes = sorted(set(indexes))
    
    return indexes


def first_zero_index(v: np.array, tol: float=1e-14):
    """
    It returns the index of the first zero element in the vector v with a certain tolerance

    Parameters
    ----------
        v: np.array
            Vector from which we want the first zero index.
        tol: float
            Tolerance below which the element is considered zero. By default, it is 1e-14.
    Returns
    ----------
    i: int
        Index of the first zero element in the vector v.
    """
    for i, val in enumerate(v):
        if abs(val) < tol:  
            return i
    return None  # If there are no zero elements, it returns None


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

    
def omega_symplectic_transformation(Omega: Matrix, no_compact_flux_variables: int, no_flux_variables: int, tol: float = 1e-14) -> tuple[Matrix, Matrix]:
    """
    Transform an antisymmetric matrix Omega to the symplectic matrix J such that J = V.T @ Omega @ V but treating the flux subspaces separetly 

    Parameters
    ----------
        Omega: array_like
            Antisymmetric matrix to which the algorithm is applied.
        no_compact_flux_variables: int
            Parameter to indicate how many compact flux variables we have.
        no_flux_variables: int
            Parameter to indicate how many total flux variables we have.
        tol: int
            Tolerance below which the element is considered zero. By default, it is 1e-14.
    
    Returns
    ----------
        J: Matrix
            Symplectic matrix from the transformation V.T @ Omega @ V
        V: Matrix
            Basis change matrix that transforms the input matrix M into J.
        no_compact_flux_variables: int
            Parameter to indicate how many independet compact flux variables we have at the end.
    """

    # Define the number of exteded flux variables and charge variables
    no_extended_flux_variables = no_flux_variables - no_compact_flux_variables
    no_charge_variables = Omega.shape[0] - no_flux_variables

    # Verify that the input matrix Omega is antisymmetric
    assert np.allclose(Omega.T, - Omega), "Input Omega matrix must be antisymmetric"
    
    # Delete the Gauge variables that already make zero their columns and rows in Omega to create the matrix Omega_new
    Omega_new = Omega.copy()
    delete_index_list = []
    for i in range(Omega.shape[0]):
        if np.all(np.abs(Omega[i, :]) < tol):
            delete_index_list.append(i)

    Omega_new = np.delete(Omega_new, delete_index_list, axis=0) # Row elimination
    Omega_new = np.delete(Omega_new, delete_index_list, axis=1) # Column elimination

    # Update the number of variables of each type according to the elimination of the previous Gauge variables
    aux1, aux2 = no_compact_flux_variables, no_flux_variables
    for _, delete_index in enumerate(delete_index_list):
        if delete_index < aux1:
            no_compact_flux_variables = no_compact_flux_variables - 1
            no_flux_variables = no_flux_variables - 1
        elif aux1 <= delete_index < aux2:
            no_extended_flux_variables = no_extended_flux_variables - 1
            no_flux_variables = no_flux_variables - 1
        elif delete_index >= aux2:
            no_charge_variables = no_charge_variables - 1 

    # Permute the variable vector in Omega to have (compact flux, charges, extended flux)
    # - Columns permutation
    Omega_perm = np.hstack((Omega_new[:, :no_compact_flux_variables], Omega_new[:, no_flux_variables:], Omega_new[:, no_compact_flux_variables:no_flux_variables]))
    # - Rows permutation
    Omega_perm = np.vstack((Omega_perm[:no_compact_flux_variables, :], Omega_perm[no_flux_variables:, :], Omega_perm[no_compact_flux_variables:no_flux_variables, :]))


    # Raise an error if there are linear dependencies between the rows of Omega
    # - Compact flux variables
    Omega_compact_flux = Omega_perm[:no_compact_flux_variables, no_compact_flux_variables:no_compact_flux_variables+no_charge_variables-no_extended_flux_variables]
    
    if len(Omega_compact_flux) > 0:
        if np.linalg.matrix_rank(Omega_compact_flux, tol=tol) < Omega_compact_flux.shape[0]:
            raise ValueError ('There are linear dependencies between the rows of Omega. By the momment the program is not ready to solve this circuit.')

    # - Extended flux variables
    Omega_extended_flux = Omega_perm[no_compact_flux_variables+no_charge_variables:, no_compact_flux_variables:no_compact_flux_variables+no_charge_variables]

    if len(Omega_extended_flux) > 0:
        if np.linalg.matrix_rank(Omega_extended_flux, tol=tol) < Omega_extended_flux.shape[0]:
            raise ValueError ('There are linear dependencies between the rows of Omega. By the momment the program is not ready to solve this circuit.')


    # Obtain the inverse of V
    inv_V = np.zeros((Omega_perm.shape[0], Omega_perm.shape[1]))
   
    # [This can be done in three lines of Numpy. Don't use for loops. Matrix structure not evident.]
    # [Here the dependent variables (zeros of the 2-form) are restored in "Omega" and "V"]
    for i in range(no_compact_flux_variables):
        inv_V[i,i] = 1
        inv_V[i + no_compact_flux_variables, :] = Omega_perm[i,:]
    
    for i in range(no_extended_flux_variables):
        inv_V[i + 2*no_compact_flux_variables, :] = Omega_perm[i + Omega_perm.shape[0] - no_extended_flux_variables,:]
 
    for i in range(no_charge_variables-no_flux_variables):
        if no_compact_flux_variables > 0: # Complete the matrix when there are compact flux variables
            inv_V[2*no_compact_flux_variables + no_extended_flux_variables:no_charge_variables + no_compact_flux_variables, no_compact_flux_variables:(no_compact_flux_variables + no_charge_variables)] = \
                null_space(inv_V[no_compact_flux_variables:2*no_compact_flux_variables, no_compact_flux_variables:((no_compact_flux_variables + no_charge_variables))]).T[i,:]
        else: # Complete the matrix when there are no compact flux variables
            inv_V[no_flux_variables:no_charge_variables, :no_charge_variables] = null_space(inv_V[:no_flux_variables, :no_charge_variables]).T[i,:]
    
    for i in range(no_extended_flux_variables):
        inv_V[i+(no_compact_flux_variables + no_charge_variables), i+(no_compact_flux_variables + no_charge_variables)] = 1

    # Permute the variable vector in inv_V to return to the initial disposition (compact flux, extended flux, compact charge, extended charge)
    # - Columns permutation
    inv_V =  np.hstack((inv_V[:, :no_compact_flux_variables], inv_V[:, no_compact_flux_variables + no_charge_variables:], inv_V[:, no_compact_flux_variables:no_charge_variables+no_compact_flux_variables]))
    # - Rows permutation
    inv_V = np.vstack((inv_V[:no_compact_flux_variables, :], inv_V[no_compact_flux_variables + no_charge_variables:, :], inv_V[no_compact_flux_variables:no_charge_variables+no_compact_flux_variables, :]))

    # Put back the columns and rows corresponding to the gauge variables deleted at the beggining
    no_gauge_variables = len(delete_index_list)
    no_non_gauge_variables = Omega_new.shape[0]
    
    if no_gauge_variables > 0:

        inv_V = np.vstack ((inv_V, np.zeros((no_gauge_variables, inv_V.shape[1]))))
        for i, delete_index in enumerate(delete_index_list): 

            inv_V = np.hstack((inv_V[:, :delete_index], np.zeros((inv_V.shape[0], 1)), inv_V[:, delete_index:]))
            inv_V[i +  no_non_gauge_variables, delete_index] = 1

    # Obtain basis change matrix V as the inverse of inv_V
    #V = pseudo_inv(inv_V, tol=tol)
    V = np.linalg.inv(inv_V)

    # Build the new 2-form and verify if it is correct
    J = np.zeros((Omega.shape[0], Omega.shape[1]))
    J[:no_flux_variables, no_flux_variables:2*no_flux_variables] = np.eye(no_flux_variables)
    J[no_flux_variables:2*no_flux_variables, :no_flux_variables] = -np.eye(no_flux_variables)

    assert np.allclose(J, V.T @ Omega @ V), 'Something goes wrong. Output matrix V must satisfy J =  V.T @ Omega @ V, with J the symplectic matrix'

    # Returns
    return J, V, no_compact_flux_variables


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
    
    # Define the symplectic matrix J with the correct dimensions
    J = np.zeros((2 * len(imag_eigval) + len(zero_eigval), 2 * len(imag_eigval) + len(zero_eigval)))
    I = np.eye(len(imag_eigval))
    
    J[:len(imag_eigval), len(imag_eigval):2 * len(imag_eigval)] = I
    J[len(imag_eigval):2 * len(imag_eigval), :len(imag_eigval)] = -I 
    
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
    T_plus = np.empty((2*len(imag_eigval) + len(zero_eigval), 0))
    T_minus = np.empty((2*len(imag_eigval) + len(zero_eigval), 0))
    T = np.empty((2*len(imag_eigval) + len(zero_eigval), 0))

    for i in range(len(imag_eigval)):
        sigma = 1j * np.sign((imag_eigvec[:,i].T @ J @ np.conj(imag_eigvec[:,i]))/1j)

        T_plus = np.hstack((T_plus, np.sqrt(2) * (normal_imag_eigvec[:,i].real).reshape(-1,1)))

        #T_minus = np.hstack((T_minus, np.sqrt(2)*((sigma * (-1) * np.conjugate(normal_imag_eigvec[:,i])).real).reshape(-1,1)))
        T_minus = np.hstack((T_minus, np.sqrt(2)*((normal_imag_eigvec[:,i]).imag).reshape(-1,1)))

    T = np.hstack((T, T_plus))
    T = np.hstack((T, T_minus))
    T = np.hstack((T, zero_eigvec))

    # Verify that the matrix T satisies the conditions it must satisfy
    assert T.shape[0] == M.shape[0], "There is an error in the construction of the normal form transfromation matrix T. \
        It must have the same dimension as the input matrix"
    assert np.allclose(J, T.T @ J @ T, rtol = tol), "There is an error in the construction of the normal form transfromation matrix T. \
        It must be symplectic, T.T @ J @ T = J"
    assert np.allclose(T.imag, 0, rtol = tol), "There is an error in the construction of the normal form transfromation matrix T. It must be real"

    # Obtain and return the output matrix 
    M_out = np.linalg.pinv(T) @ M @ T

    return M_out, T