import numpy as np
def lu_factorization(A_in):
    A = A_in.astype(float).copy()
    n = A.shape[0]
    L, U = np.zeros_like(A), np.zeros_like(A)

    for k in range(n):
        i, j = k, k
        if np.isclose(A[i, j], 0.0):        
            # Search for the first shell `i` (relative to k) 
            # that contains a non-zero element in either its row or column slice.
            is_nz = ~np.isclose(A[k:, k:], 0.0)
            
            # valid_rows[x] is True if Row x of A[k:, k:] (upper part, j>=x) has non-zero
            valid_rows = np.any(np.triu(is_nz), axis=1)
            # valid_cols[x] is True if Col x of A[k:, k:] (lower part, i>=x) has non-zero
            valid_cols = np.any(np.tril(is_nz), axis=0)
            
            valid_shells = valid_rows | valid_cols
            
            if np.any(valid_shells):
                l_rel = np.argmax(valid_shells) # First index where valid
                l = k + l_rel                
                # Now find k_row and k_col for this specific shell l                
                k_row = l + np.argmax(~np.isclose(A[l, l:], 0.0)) if valid_rows[l_rel] else n + 1
                k_col = l + np.argmax(~np.isclose(A[l:, l], 0.0)) if valid_cols[l_rel] else n + 1                
                use_row = k_row <= k_col
                if l == k and np.any(~np.isclose(A[k:, k] if use_row else A[k, k:], 0.0)):
                    print("Error: Pivot 0, but both row and col non-zero.")
                    return None         
                i, j = (l, k_row) if use_row else (k_col, l)
            else:
                break

        L[k:, k] = A[k:, j] / A[i, j]
        U[k, k:] = A[i, k:]
        A[k+1:, k+1:] -= np.outer(L[k+1:, k], U[k, k+1:])
    
    return L, U

if __name__ == "__main__":
    np.random.seed(42)
    n_tests = 1000; p = 0.2; n = 16; success = 0
    for _ in range(n_tests):
        L_gen = np.tril((np.random.rand(n,n)<p).astype(float))
        U_gen = np.triu((np.random.rand(n,n)<p).astype(float))
        A_rand = L_gen @ U_gen
        res = lu_factorization(A_rand)
        if (res and np.allclose(A_rand, res[0] @ res[1]) 
            and np.allclose(res[0], np.tril(res[0])) 
            and np.allclose(res[1], np.triu(res[1]))): success += 1
    print(f"PASS: {success}/{n_tests} tests passed.")
