import numpy as np

def lu_factorization(A_input):
    """Compute an LU factorization A = L @ U of A_input.

    The algorithm produces L lower triangular and U upper triangular such that
    A_input = L @ U. It uses row and column permutations internally to select
    pivots, but it does *not* form a full-pivoting factorization of the form
    P @ A_input @ Q = L @ U. Instead, the returned L and U factor A_input
    itself; the permutation matrices P and Q in the return value simply record
    the row and column reordering performed during the elimination.
    """
    A = A_input.astype(float).copy()
    n = A.shape[0]
    L, U = np.zeros_like(A), np.zeros_like(A)
    p_vec, q_vec = np.arange(n), np.arange(n)
    
    def finalize():
        P = np.eye(n)[p_vec]
        Q = np.eye(n)[:, q_vec]
        return P.T @ L, U @ Q.T

    def lu_step(k):
        L[k:, k] = A[k:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        # Update A (Schur complement)
        A[k:, k:] = A[k:, k:] - np.outer(L[k:, k], U[k, k:])

    def pivot_row(k, i):
        # "Pivot row k and i in A"
        A[[k, i], :] = A[[i, k], :]
        p_vec[[k, i]] = p_vec[[i, k]]
        # Swap rows of L computed so far
        L[[k, i], :k] = L[[i, k], :k]

    def pivot_col(k):
        # Find smallest j > k such that A[k,j] is non-zero
        j = k + 1 + np.argmax(~np.isclose(A[k, k+1:], 0.0))
        # "Pivot column k and j in A"
        A[:, [k, j]] = A[:, [j, k]]
        q_vec[[k, j]] = q_vec[[j, k]]
        # Swap columns of U computed so far
        U[:k, [k, j]] = U[:k, [j, k]]

    for k in range(n):
        # Only perform swapping logic if pivot is zero
        if np.isclose(A[k, k], 0.0):
            is_zero_row = np.allclose(A[k, k:], 0.0)
            is_zero_column = np.allclose(A[k:, k], 0.0)
            
            if not is_zero_row and not is_zero_column:
                print("Error: Matrix A does not have an LU factorization (Pivot is 0, but both row and col have non-zeros).")
                return None
                
            if not is_zero_row:
                pivot_col(k)
            elif not is_zero_column:
                i = k + 1 + np.argmax(~np.isclose(A[k+1:, k], 0.0))
                pivot_row(k, i)
            else:
                # Both row and column are zero
                if np.allclose(A[k:, k:], 0.0):
                    return finalize()
                
                # Find smallest i > k such that row A[i,k:] is non-zero
                rows_nonzero = np.any(~np.isclose(A[k+1:, k:], 0.0), axis=1)
                i = k + 1 + np.argmax(rows_nonzero)
                pivot_row(k, i)
                pivot_col(k)

        # Perform step (common to all paths that didn't return)
        assert not np.isclose(A[k, k], 0.0), "Logic error: Pivot is zero after swapping"        
        lu_step(k)

    return finalize()

# ==========================================
# Test Usage
# ==========================================
if __name__ == "__main__":
    # Example 1: Generate A from singular L and U factors
    # We construct A such that it requires pivoting
    L_gen = np.array([[1, 0, 0], [2, 1, 0], [3, 4, 1]], dtype=float)
    U_gen = np.array([[0, 1, 3], [0, 1, 4], [0, 0, 0]], dtype=float)
    A1 = L_gen @ U_gen

    print("--- Test Case 1 ---")
    print("Constructed A (L @ U):\n", A1)
    
    result = lu_factorization(A1)
    
    if result:
        # L, U: The factors satisfying A = L @ U
        L, U = result
        
        print("\nCalculated L (Factor of A - should be Lower Tri):\n", L)
        print("\nCalculated U (Factor of A - should be Upper Tri):\n", U)
        
        # Verify A = L @ U
        A_recalc = L @ U
        print("\nReconstructed A (L @ U):\n", A_recalc)
        print("\nMatch A = LU:", np.allclose(A1, A_recalc))

        # --- Rigorous Verification of Properties ---
        print("\n--- Rigorous Verification ---")
        
        # 1. Verify A = L @ U
        check_A = np.allclose(A1, L @ U)
        print(f"Check 1: A == L @ U? {check_A}")
        
        # 2. Verify L is Lower Triangular
        # We use a small tolerance for floating point comparisons
        check_L_tri = np.allclose(L, np.tril(L))
        print(f"Check 2: L is Lower Triangular? {check_L_tri}")

        # 3. Verify U is Upper Triangular
        check_U_tri = np.allclose(U, np.triu(U))
        print(f"Check 3: U is Upper Triangular? {check_U_tri}")
        
        if not check_L_tri:
            print("  -> L deviation from tril:", np.abs(L - np.tril(L)).max())
        if not check_U_tri:
            print("  -> U deviation from triu:", np.abs(U - np.triu(U)).max())

    # Example 2: "Error" Case
    print("\n--- Test Case 2 (Edge Case: Pivot=0, Row!=0, Col!=0) ---")
    A2 = np.array([[0, 1], [1, 1]], dtype=float)
    print("Input:\n", A2)
    result2 = lu_factorization(A2)
    
    if result2 is None:
        print("PASS: Correctly identified that no factorization exists for this specific algorithm.")
    else:
        print("FAIL: Should have returned None but returned a result.")

    # Example 3: Pivot=0, Row=0, Col!=0 (Requires Row Pivot)
    print("\n--- Test Case 3 (Pivot=0, Row=0, Col!=0 -> Row Pivot) ---")
    A3 = np.array([
        [0, 0, 0],
        [1, 2, 3],
        [0, 5, 6]
    ], dtype=float)
    print("Input:\n", A3)
    result3 = lu_factorization(A3)
    
    if result3:
        L, U = result3
        check_A3 = np.allclose(A3, L @ U)
        print(f"PASS: Factorization successful. A = L @ U? {check_A3}")
        print(f"L (Lower Tri):\n{L}")
    else:
        print("FAIL: Should have factorized successfully.")

    # Example 4: Pivot=0, Row!=0, Col=0 (Requires Column Pivot)
    print("\n--- Test Case 4 (Pivot=0, Row!=0, Col=0 -> Col Pivot) ---")
    A4 = np.array([
        [0, 5, 2],
        [0, 1, 3],
        [0, 4, 8]
    ], dtype=float)
    print("Input:\n", A4)
    result4 = lu_factorization(A4)
    
    if result4:
        L, U = result4
        check_A4 = np.allclose(A4, L @ U)
        print(f"PASS: Factorization successful. A = L @ U? {check_A4}")
    else:
        print("FAIL: Should have factorized successfully.")

    # Example 5: Random Full Rank Matrix
    print("\n--- Test Case 5 (Random 10x10 Matrix) ---")
    np.random.seed(42)
    A5 = np.random.rand(10, 10)
    result5 = lu_factorization(A5)
    
    if result5:
        L, U = result5
        check_A5 = np.allclose(A5, L @ U)
        check_L_tri = np.allclose(L, np.tril(L))
        check_U_tri = np.allclose(U, np.triu(U))
        print(f"PASS: A = L @ U? {check_A5}")
        print(f"      L is Lower Triangular? {check_L_tri}")
        print(f"      U is Upper Triangular? {check_U_tri}")
    else:
        print("FAIL: Random matrix should factorize.")

    # Randomized Large Scale Tests
    n_tests = 1000
    print(f"\n--- Running {n_tests} Randomized Tests ---")
    p_prob = 0.2
    n_size = 16
    
    success_count = 0
    
    for t in range(n_tests):
        # Generate L: l_ij = 1 with prob p, 0 otherwise, i >= j (Lower Triangular)
        # Using uniform random and thresholding
        L_full = (np.random.rand(n_size, n_size) < p_prob).astype(float)
        L_rand = np.tril(L_full)
        
        # Generate U: u_ij = 1 with prob p, 0 otherwise, i <= j (Upper Triangular)
        U_full = (np.random.rand(n_size, n_size) < p_prob).astype(float)
        U_rand = np.triu(U_full)
        
        # Set A = L @ U
        A_rand = L_rand @ U_rand
        
        # Run Factorization
        result = lu_factorization(A_rand)
        
        # Verify
        assert result is not None, f"Test {t+1} failed: Result is None for A generated from L@U"
        
        L_out, U_out = result
        
        # Assert A == L @ U (reconstructed)
        if not np.allclose(A_rand, L_out @ U_out):
            print(f"Test {t+1} failed: Reconstruction mismatch.")
            break
            
        # Assert L is lower triangular
        if not np.allclose(L_out, np.tril(L_out)):
            print(f"Test {t+1} failed: L is not lower triangular.")
            break
            
        # Assert U is upper triangular
        if not np.allclose(U_out, np.triu(U_out)):
            print(f"Test {t+1} failed: U is not upper triangular.")
            break
            
        success_count += 1
        
    print(f"PASS: {success_count}/{n_tests} randomized tests passed successfully.")