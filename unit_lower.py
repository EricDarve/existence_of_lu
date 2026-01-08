import numpy as np
from algo2 import unit_lower_lu_factorization

if __name__ == "__main__":
    np.random.seed(42)
    # 1. Random Matrix
    A1 = np.random.rand(10, 10)
    res1 = unit_lower_lu_factorization(A1)
    if res1:
        print(f"Test 1 (Random):\n  A=LU? {np.allclose(A1, res1[0] @ res1[1])}")
        print(f"  Unit Lower? {np.allclose(res1[0], np.tril(res1[0])) and np.allclose(np.diag(res1[0]), 1)}")
        print(f"  U Upper? {np.allclose(res1[1], np.triu(res1[1]))}")
    else:
        print("Test 1 (Random): Failed")

    # 2. Impossible Case (Pivot=0, Col!=0)
    A2 = np.array([[0,0], [1,0]])
    print("\nTest 2 (Impossible)")
    if unit_lower_lu_factorization(A2) is None: print("PASS: Correctly returned None.")

    # 3. Pivot Needed (Pivot=0, Row!=0, Col=0) -> Swap Col
    A3 = np.array([[0, 2], [0, 4]]) 
    res3 = unit_lower_lu_factorization(A3)
    if res3:
        print(f"\nTest 3 (Pivot): A=LU? {np.allclose(A3, res3[0] @ res3[1])}")
    else:
        print("Test 3 (Pivot): Failed")
    
    # 4. Singular Block (Pivot=0, Row=0, Col=0)
    A4 = np.array([[0,0], [0,1]])
    res4 = unit_lower_lu_factorization(A4)
    if res4:
         print(f"Test 4 (Singular): A=LU? {np.allclose(A4, res4[0] @ res4[1])}")
    else: 
        print("Test 4 (Singular): Failed")

    # Randomized Large Scale Tests
    print(f"\nRunning Randomized Tests")
    n_tests = 20
    p = 0.1
    n = 100

    success = 0
    for _ in range(n_tests):
        L_gen = np.tril(np.random.rand(n,n)<p, -1) + np.eye(n)
        U_gen = np.triu(np.random.rand(n,n)<p)        
        # Choose a random column between 1 and n-2
        j = np.random.randint(0, n-1)
        L_gen[j, j] = 0
        L_gen[j+1:, j] = 1 + np.random.rand(n-j-1)
        U_gen[:j+1, :j+1] = np.eye(j+1) + np.triu(np.random.rand(j+1,j+1), 1)
        A_rand = L_gen @ U_gen
        res = unit_lower_lu_factorization(A_rand)
        if (res is None): success += 1            
    print(f"Impossible Cases: {success}/{n_tests} tests passed.")    

    success = 0    
    for _ in range(n_tests):
        L_gen = np.tril(np.random.rand(n,n)<p, -1) + np.eye(n)
        U_gen = np.triu(np.random.rand(n,n)<p)
        A_rand = L_gen @ U_gen
        res = unit_lower_lu_factorization(A_rand)
        if (res 
            and np.allclose(A_rand, res[0] @ res[1])
            and np.allclose(res[0], np.tril(res[0])) 
            and np.allclose(np.diag(res[0]), 1.0)
            and np.allclose(res[1], np.triu(res[1]))): success += 1
    print(f"Successful Cases: {success}/{n_tests} tests passed.")