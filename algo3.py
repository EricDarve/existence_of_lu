import numpy as np
def lu_pivoting(A_in):
    A = A_in.astype(np.float64).copy()
    n = A.shape[0]
    L, U = np.zeros_like(A), np.zeros_like(A)
    p, q = np.arange(n), np.arange(n)

    def swap_rows(k, i):
        A[[k, i], k:] = A[[i, k], k:]
        p[[k, i]] = p[[i, k]]
        L[[k, i], :k] = L[[i, k], :k]

    def swap_cols(k, j):
        A[k:, [k, j]] = A[k:, [j, k]]
        q[[k, j]] = q[[j, k]]
        U[:k, [k, j]] = U[:k, [j, k]]

    for k in range(n):
        idx_max = np.argmax(np.abs(A[k:, k:]))
        i_local, j_local = np.unravel_index(idx_max, A[k:, k:].shape)
        i_max, j_max = k + i_local, k + j_local
        swap_rows(k, i_max)
        swap_cols(k, j_max)
        if np.isclose(A[k, k], 0.0): break
        
        L[k:, k] = A[k:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        A[k+1:, k+1:] -= np.outer(L[k+1:, k], U[k, k+1:])

    P, Q = np.eye(n)[p], np.eye(n)[:, q]
    return P, Q, L, U