import numpy as np
def unit_lower_lu_factorization(A_in):
    A = A_in.astype(float).copy()
    n = A.shape[0]
    L, U = np.eye(n), np.zeros_like(A)
    q = np.arange(n)

    for k in range(n):
        j = k + np.argmax(~np.isclose(A[k, k:], 0.0))
        if np.isclose(A[k, k], 0.0):
            if not np.allclose(A[k+1:, k], 0.0):
                print(f"Error: Pivot 0, but col non-zero.")
                return None
            if np.isclose(A[k, j], 0.0): continue

        L[k+1:, k] = A[k+1:, j] / A[k, j]
        U[k, k:] = A[k, k:]
        A[k+1:, k+1:] -= np.outer(L[k+1:, k], U[k, k+1:])

    Q = np.eye(n)[:, q]
    return L, U @ Q.T

if __name__ == "__main__":
    np.random.seed(42)
    n_tests = 1000; p = 0.2; n = 16; success = 0
    for _ in range(n_tests):
        L_gen = np.tril(np.random.rand(n,n)<p, -1).astype(float) + np.eye(n)
        U_gen = np.triu(np.random.rand(n,n)<p).astype(float)
        A_rand = L_gen @ U_gen
        res = unit_lower_lu_factorization(A_rand)
        if (res 
            and np.allclose(A_rand, res[0] @ res[1])
            and np.allclose(res[0], np.tril(res[0])) 
            and np.allclose(np.diag(res[0]), 1.0)
            and np.allclose(res[1], np.triu(res[1]))): success += 1
    print(f"PASS: {success}/{n_tests} tests passed.")      
