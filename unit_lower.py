import numpy as np

def unit_lower_lu_factorization(A_in):
    """
    Computes Unit Lower LU factorization A = L @ U using internal column permutations.
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
    np.random.seed(42)
    # 1. Random Matrix
    A1 = np.random.rand(10, 10)
    res1 = unit_lower_lu_factorization(A1)
    if res1:
        print(f"Test 1 (Random): A=LU? {np.allclose(A1, res1[0] @ res1[1])}")
        print(f"  Unit Lower? {np.allclose(res1[0], np.tril(res1[0])) and np.allclose(np.diag(res1[0]), 1)}")
        print(f"  U Upper? {np.allclose(res1[1], np.triu(res1[1]))}")

    # 2. Impossible Case (Pivot=0, Col!=0)
    A2 = np.array([[0,0], [1,0]], dtype=float)
    print("\n--- Test 2 (Impossible) ---")
    if unit_lower_lu_factorization(A2) is None: print("PASS: Correctly returned None.")

    # 3. Pivot Needed (Pivot=0, Row!=0, Col=0) -> Swap Col
    A3 = np.array([[0, 2], [0, 4]], dtype=float) 
    res3 = unit_lower_lu_factorization(A3)
    if res3:
        print(f"\nTest 3 (Pivot): A=LU? {np.allclose(A3, res3[0] @ res3[1])}")
    
    # 4. Singular Block (Pivot=0, Row=0, Col=0)
    A4 = np.array([[0,0], [0,1]], dtype=float)
    res4 = unit_lower_lu_factorization(A4)
    if res4:
         print(f"Test 4 (Singular): A=LU? {np.allclose(A4, res4[0] @ res4[1])}")
    
    # Randomized Large Scale Tests
    print(f"\n--- Running 1000 Randomized Tests ---")
    n_tests = 1000
    n = 20
    success = 0
    for _ in range(n_tests):
        L_gen = np.tril(np.random.rand(n,n), -1) + np.eye(n)
        U_gen = np.triu(np.random.rand(n,n))
        A_rand = L_gen @ U_gen
        res = unit_lower_lu_factorization(A_rand)
        if (res and np.allclose(A_rand, res[0] @ res[1])
            and np.allclose(res[0], np.tril(res[0])) 
            and np.allclose(np.diag(res[0]), 1.0)
            and np.allclose(res[1], np.triu(res[1]))): success += 1
    print(f"PASS: {success}/{n_tests} tests passed.")
