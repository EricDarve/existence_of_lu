from algo1 import lu_factorization

import numpy as np
if __name__ == "__main__":

    # Test suite
    np.random.seed(42)
    
    # 1. Random Matrix
    # Product of a random lower and upper triangular 
    # matrix with entries in [0,1] should pass
    print("\nTest 1 (Random):")
    L = np.tril(np.random.rand(10, 10))
    U = np.triu(np.random.rand(10, 10))
    A1 = L @ U
    res1 = lu_factorization(A1)
    if res1: 
        print(f"  A=LU? {np.allclose(A1, res1[0] @ res1[1])}")
        print(f"  L Lower? {np.allclose(res1[0], np.tril(res1[0]))}")
        print(f"  U Upper? {np.allclose(res1[1], np.triu(res1[1]))}")
    else:
        print("  FAIL.")        

    # 2. Error Case (Row!=0 Col!=0)
    A2 = np.array([[0., 1.], [1., 1.]])
    # Code should ERROR per logic: "if not row_zero and not col_zero: Error"
    print("\nTest 2 (Error Case)")
    res2 = lu_factorization(A2)
    if res2 is None: 
        print("PASS: Correctly returned None.")
    else:
        print(A2)
        print(res2[0])
        print(res2[1])
        print(res2[0]@res2[1])
        print("  FAIL.")

    # 3. Pivot Needed (Row=0, Col!=0) -> Swap Row
    A3 = np.array([[0,0,0], [1,2,3], [0,5,6]], dtype=float)
    print(f"\nTest 3 (Row Swap):")
    res3 = lu_factorization(A3)
    if res3: 
        print(f"  A=LU? {np.allclose(A3, res3[0] @ res3[1])}")
    else:
        print("  FAIL.")

    # 4. Pivot Needed (Row!=0, Col=0) -> Swap Col
    A4 = np.array([[0,5,2], [0,1,3], [0,4,8]], dtype=float)
    print(f"\nTest 4 (Col Swap):")
    res4 = lu_factorization(A4)
    if res4:
        print(f"  A=LU? {np.allclose(A4, res4[0] @ res4[1])}")
    else:
        print("  FAIL.")    
    
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
    print(f"\nTest 5 (Singular Block):")
    res5 = lu_factorization(A5)
    if res5:
        print(f"  A=LU? {np.allclose(A5, res5[0] @ res5[1])}")
        print(f"  L Lower? {np.allclose(res5[0], np.tril(res5[0]))}")
        print(f"  U Upper? {np.allclose(res5[1], np.triu(res5[1]))}")
    else:
        print("  FAIL.")

    # Randomized Large Scale
    print(f"\nRunning Randomized Tests")
    n_tests = 20
    p = 0.99
    n = 1000

    # Failing tests
    success = 0    
    for _ in range(n_tests):
        r = np.random.randint(1, n-1)
        L_gen = np.tril((np.random.rand(n,r)<p).astype(float)) + np.eye(n,r)
        U_gen = np.triu((np.random.rand(r,n)<p).astype(float)) + np.eye(r,n)
        A_rand = L_gen @ U_gen
        # Create a random matrix of size n-r
        A_botright = 1 + np.random.rand(n-r, n-r)
        A_botright[0,0] = 0
        # Add to the bottom right part of A_rand
        A_rand[r:, r:] += A_botright
        res = lu_factorization(A_rand)
        # This should fail
        if res is None: success += 1
        else: print("FAIL.")

    print(f"Impossible Cases: {success}/{n_tests} tests passed.")  

    success = 0
    import time
    start_time = time.time()
    for _ in range(n_tests):
        L_gen = np.tril((np.random.rand(n, n) < p).astype(float))
        U_gen = np.triu((np.random.rand(n, n) < p).astype(float))
        A_rand = L_gen @ U_gen
        res = lu_factorization(A_rand)
        if (res and np.allclose(A_rand, res[0] @ res[1]) 
            and np.allclose(res[0], np.tril(res[0])) 
            and np.allclose(res[1], np.triu(res[1]))): success += 1
        else:
            # Print diagnostics
            print("FAIL.")
            print(A_rand)
    end_time = time.time()
    print(f"Successful Cases: {success}/{n_tests} tests passed.")
    print(f"  Time taken for {n_tests} tests: {end_time - start_time:.4f} seconds")    