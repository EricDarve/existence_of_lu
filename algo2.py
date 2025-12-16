import numpy as np

def unit_lower_lu_factorization(A_in):
    """
    Computes Unit Lower LU factorization A = L @ U.
    """
    A = A_in.astype(float).copy()
    n = A.shape[0]
    L, U = np.eye(n), np.zeros_like(A)
    q = np.arange(n)

    def swap_cols(k, j):
        A[:, [k, j]] = A[:, [j, k]]
        q[[k, j]] = q[[j, k]]
        U[:k, [k, j]] = U[:k, [j, k]]

    for k in range(n):
        if np.isclose(A[k, k], 0.0):
            if not np.allclose(A[k+1:, k], 0.0):
                print(f"Error: Zero pivot at k={k} but column {k} not zero.")
                return None
            
            if not np.allclose(A[k, k+1:], 0.0):
                j = k + 1 + np.argmax(~np.isclose(A[k, k+1:], 0.0))
                swap_cols(k, j)
            else:
                continue

        L[k+1:, k] = A[k+1:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        A[k+1:, k+1:] -= np.outer(L[k+1:, k], U[k, k+1:])

    Q = np.eye(n)[:, q]
    return L, U @ Q.T


if __name__ == "__main__":
    A = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    print("A =\n", A)
    L, U = unit_lower_lu_factorization(A)
    if L is not None and U is not None: 
        print(f"A = LU? {np.allclose(A, L @ U)}")
        print(f"L Unit Lower? {np.allclose(L, np.tril(L)) and np.allclose(np.diag(L), 1.0)}")
        print(f"U Upper? {np.allclose(U, np.triu(U))}")        
