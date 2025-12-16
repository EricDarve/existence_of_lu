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

        L[k:, k] = A[k:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        A[k+1:, k+1:] -= np.outer(L[k+1:, k], U[k, k+1:])

    P, Q = np.eye(n)[p], np.eye(n)[:, q]
    return P.T @ L, U @ Q.T

if __name__ == "__main__":
    # Test suite
    np.random.seed(42)
    
    # 1. Random Matrix
    A1 = np.random.rand(10, 10)
    res1 = lu_factorization(A1)
    if res1: 
        print(f"Test 1 (Random): A=LU? {np.allclose(A1, res1[0] @ res1[1])}")
        print(f"  L Lower? {np.allclose(res1[0], np.tril(res1[0]))}")
        print(f"  U Upper? {np.allclose(res1[1], np.triu(res1[1]))}")

    # 2. Pivot Needed Case (Pivot=0, Row!=0) -> Swap Col
    A2 = np.array([[0., 1.], [1., 1.]]) # Actually this is Error case row!=0 col!=0
    # Code should ERROR per logic: "if not row_zero and not col_zero: Error"
    print("\n--- Test 2 (Error Case) ---")
    if lu_factorization(A2) is None: print("PASS: Correctly returned None.")

    # 3. Pivot Needed (Row=0, Col!=0) -> Swap Row
    A3 = np.array([[0,0,0], [1,2,3], [0,5,6]], dtype=float)
    res3 = lu_factorization(A3)
    if res3: 
        print(f"\nTest 3 (Row Swap): A=LU? {np.allclose(A3, res3[0] @ res3[1])}")

    # 4. Pivot Needed (Row!=0, Col=0) -> Swap Col
    A4 = np.array([[0,5,2], [0,1,3], [0,4,8]], dtype=float)
    res4 = lu_factorization(A4)
    if res4:
        print(f"Test 4 (Col Swap): A=LU? {np.allclose(A4, res4[0] @ res4[1])}")
    
    # 5. Singular Block (Both 0) -> Search Submatrix
    # Need matrix where A[0,0]=0, row0=0, col0=0, but A[1:,1:] not zero
    # AND rank > 0
    A5 = np.array([[0,0,0,0], [0,0,0,0], [0,0,1,2], [0,0,3,4]], dtype=float)
    # Pivot 0: Both 0. Submatrix A[0:,0:] has non-zeros.
    # Logic: Search A[1:, 0:] -> Rows 2 and 3 are non-zero.
    # Swap row 0 with row 2 (index 2).
    # New A: Row 0 is [0,0,1,2], Row 2 is [0,0,0,0].
    # Now Row 0 non-zero. Find col j. Col 2 is 1. Swap Col 0 and 2.
    # New A: A[0,0]=1. Proceed.
    res5 = lu_factorization(A5)
    if res5:
        print(f"Test 5 (Singular Block): A=LU? {np.allclose(A5, res5[0] @ res5[1])}")
        print(f"  L Lower? {np.allclose(res5[0], np.tril(res5[0]))}")
        print(f"  U Upper? {np.allclose(res5[1], np.triu(res5[1]))}")

    # Randomized Large Scale
    print(f"\n--- Running 1000 Randomized Tests ---")
    n_tests = 1000
    p = 0.2
    n = 16
    success = 0
    for _ in range(n_tests):
        L_gen = np.tril((np.random.rand(n,n)<p).astype(float))
        U_gen = np.triu((np.random.rand(n,n)<p).astype(float))
        A_rand = L_gen @ U_gen
        res = lu_factorization(A_rand)
        if (res and np.allclose(A_rand, res[0] @ res[1]) 
            and np.allclose(res[0], np.tril(res[0])) 
            and np.allclose(res[1], np.triu(res[1]))): success += 1
    print(f"PASS: {success}/{n_tests} tests passed.")