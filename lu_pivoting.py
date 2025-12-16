import numpy as np

def lu_pivoting(A_in):
    """
    Computes LU factorization P @ A @ Q = L @ U using full pivoting.
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
        if np.allclose(A[k:, k:], 0.0): break

        idx_max = np.argmax(np.abs(A[k:, k:]))
        i_local, j_local = np.unravel_index(idx_max, A[k:, k:].shape)
        i_max, j_max = k + i_local, k + j_local

        swap_rows(k, i_max)
        swap_cols(k, j_max)

        L[k:, k] = A[k:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        A[k+1:, k+1:] -= np.outer(L[k+1:, k], U[k, k+1:])

    P, Q = np.eye(n)[p], np.eye(n)[:, q]
    return P, Q, L, U

if __name__ == "__main__":
    np.random.seed(42)    
    print(f"\n--- Running Randomized Tests ---")
    n_tests = 1000
    n = 16
    success = 0
    for _ in range(n_tests):
        A_rand = np.random.rand(n,n)
        res = lu_pivoting(A_rand)
        if (res and np.allclose(res[0] @ A_rand @ res[1], res[2] @ res[3]) 
            and np.allclose(res[2], np.tril(res[2])) 
            and np.allclose(res[3], np.triu(res[3]))): success += 1
    print(f"PASS: {success}/{n_tests} tests passed.")