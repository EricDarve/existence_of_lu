import numpy as np
def unit_lower_lu_factorization(A_in):
    A = A_in.astype(np.float64).copy()
    n = A.shape[0]
    L, U = np.eye(n), np.zeros_like(A)

    for k in range(n):
        j = k + np.argmax(~np.isclose(A[k, k:], 0.0))
        if np.isclose(A[k, k], 0.0):
            if not np.allclose(A[k+1:, k], 0.0):
                # Factorization does not exist
                return None
            if np.isclose(A[k, j], 0.0): continue

        L[k+1:, k] = A[k+1:, j] / A[k, j]
        U[k, k:] = A[k, k:]
        A[k+1:, k+1:] -= np.outer(L[k+1:, k], U[k, k+1:])
    
    return L, U