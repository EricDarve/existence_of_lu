import numpy as np

def lu_factorization(A_in):
    """
    Computes LU factorization A = L @ U using internal row and column permutations.
    """
    A = A_in.astype(float).copy()
    n = A.shape[0]
    L, U = np.zeros_like(A), np.zeros_like(A)
    p, q = np.arange(n), np.arange(n)

    def swap_rows(k, i):
        A[[k, i], :] = A[[i, k], :]
        p[[k, i]] = p[[i, k]]
        L[[k, i], :k] = L[[i, k], :k]

    def swap_cols(k, j):
        A[:, [k, j]] = A[:, [j, k]]
        q[[k, j]] = q[[j, k]]
        U[:k, [k, j]] = U[:k, [j, k]]

    for k in range(n):
        if np.isclose(A[k, k], 0.0):
            row_zero = np.allclose(A[k, k:], 0.0)
            col_zero = np.allclose(A[k:, k], 0.0)

            if not row_zero and not col_zero:
                print("Error: Pivot 0, but both row and col non-zero. Impossible.")
                return None
            
            if not row_zero:
                j = k + 1 + np.argmax(~np.isclose(A[k, k+1:], 0.0))
                swap_cols(k, j)
            elif not col_zero:
                i = k + 1 + np.argmax(~np.isclose(A[k+1:, k], 0.0))
                swap_rows(k, i)
            else: # Both zero
                if np.allclose(A[k:, k:], 0.0): break
                # Find row i > k with non-zero entry in A[i, k:]
                i = k + 1 + np.argmax(np.any(~np.isclose(A[k+1:, k:], 0.0), axis=1))
                swap_rows(k, i)
                # Now row k is non-zero, find col j
                j = k + 1 + np.argmax(~np.isclose(A[k, k+1:], 0.0))
                swap_cols(k, j)

        L[k, k] = 1.0
        L[k:, k] = A[k:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        A[k+1:, k+1:] -= np.outer(L[k+1:, k], U[k, k+1:])

    P, Q = np.eye(n)[p], np.eye(n)[:, q]
    return P.T @ L, U @ Q.T


if __name__ == "__main__":
    A = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    print("A =\n", A)
    L, U = lu_factorization(A)
    if L is not None and U is not None: 
        print(f"A = LU? {np.allclose(A, L @ U)}")
        print(f"L Lower? {np.allclose(L, np.tril(L))}")
        print(f"U Upper? {np.allclose(U, np.triu(U))}")
