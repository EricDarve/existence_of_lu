import numpy as np


def lu_factorization(A_input):
    """Compute an LU factorization A = L @ U of A_input.

    The algorithm produces L lower triangular and U upper triangular such that
    A_input = L @ U. It uses row and column permutations internally to select
    pivots, but it does *not* form a full-pivoting factorization of the form
    P @ A_input @ Q = L @ U. Instead, the returned L and U factor A_input
    itself; the permutation matrices P and Q simply record the row and column
    reordering performed during the elimination.
    """
    A = A_input.astype(float).copy()
    n = A.shape[0]
    L, U = np.zeros_like(A), np.zeros_like(A)
    p_vec, q_vec = np.arange(n), np.arange(n)

    def finalize():
        P, Q = np.eye(n)[p_vec], np.eye(n)[:, q_vec]
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
        j = k + 1 + np.argmax(~np.isclose(A[k, k + 1 :], 0.0))
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
                print(
                    "Error: Matrix A does not have an LU factorization (Pivot is 0, but both row and col have non-zeros)."
                )
                return None

            if not is_zero_row:
                pivot_col(k)
            elif not is_zero_column:
                i = k + 1 + np.argmax(~np.isclose(A[k + 1 :, k], 0.0))
                pivot_row(k, i)
            else:
                # Both row and column are zero
                if np.allclose(A[k:, k:], 0.0):
                    return finalize()

                # Find smallest i > k such that row A[i,k:] is non-zero
                rows_nonzero = np.any(~np.isclose(A[k + 1 :, k:], 0.0), axis=1)
                i = k + 1 + np.argmax(rows_nonzero)
                pivot_row(k, i)
                pivot_col(k)

        # Perform step (common to all paths that didn't return)
        assert not np.isclose(A[k, k], 0.0), "Logic error: Pivot is zero"
        lu_step(k)

    return finalize()


if __name__ == "__main__":
    A = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    print("A =\n", A)
    L, U = lu_factorization(A)
    assert np.allclose(L, np.tril(L)), "L is not Lower Triangular"
    assert np.allclose(U, np.triu(U)), "U is not Upper Triangular"
    assert np.allclose(A, L @ U), "A != L @ U"
    print("PASS: LU factorization successful.")
