import numpy as np
def lu_factorization(A_in):
    A = A_in.astype(np.float64).copy()
    n = A.shape[0]
    L, U = np.zeros_like(A), np.zeros_like(A)
    for k in range(n):
        i, j = k, k
        if np.isclose(A[i, j], 0.0):        
            # Search for the first shell `i` (relative to k) 
            # that contains a non-zero element in either its row or column slice.
            is_nz = ~np.isclose(A[k:, k:], 0.0)
            
            # valid_rows[x] is True if Row x of 
            # A[k:, k:] (upper part, j>=x) has non-zero
            valid_rows = np.any(np.triu(is_nz), axis=1)
            # valid_cols[x] is True if Col x of 
            # A[k:, k:] (lower part, i>=x) has non-zero
            valid_cols = np.any(np.tril(is_nz), axis=0)    
            valid_shells = valid_rows | valid_cols     

            if np.any(valid_shells):
                l_rel = np.argmax(valid_shells) # First index where valid
                l = k + l_rel                
                # Find k_row and k_col for this specific shell l                
                k_row = l + np.argmax(~np.isclose(A[l, l:], 0.0)) if valid_rows[l_rel] else n + 1
                k_col = l + np.argmax(~np.isclose(A[l:, l], 0.0)) if valid_cols[l_rel] else n + 1                
                use_row = k_row <= k_col
                if l == k and np.any(~np.isclose(A[k:, k] if use_row else A[k, k:], 0.0)):
                    # Factorization does not exist
                    return None         
                i, j = (l, k_row) if use_row else (k_col, l)
            else:
                break

        L[k:, k] = A[k:, j] / A[i, j]
        U[k, k:] = A[i, k:]
        A[k+1:, k+1:] -= np.outer(L[k+1:, k], U[k, k+1:])
    
    return L, U