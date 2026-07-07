import numpy as np
import itertools

rng = np.random.default_rng(0)

def rank(M):
    # exact integer/rational rank over Q via fraction-free Gaussian elimination on floats
    # We use a robust SVD rank for verification over the reals; entries are small integers
    # so float rank is reliable. For partial permutations entries are 0/1, fully exact.
    if M.size == 0:
        return 0
    return np.linalg.matrix_rank(M)

def nullity_right(M):
    # right nullity = ncols - rank
    return M.shape[1] - rank(M)

def random_partial_perm(n, density=0.6):
    # build a partial permutation: choose a random injective partial map cols->rows
    P = np.zeros((n, n), dtype=int)
    cols = list(range(n))
    rng.shuffle(cols)
    rows_used = set()
    for j in cols:
        if rng.random() < density:
            avail = [r for r in range(n) if r not in rows_used]
            if not avail:
                continue
            i = avail[rng.integers(len(avail))]
            P[i, j] = 1
            rows_used.add(i)
    return P

def combinatorial_quantities(P):
    n = P.shape[0]
    # matched pairs: P[i,j]=1
    matches = [(i, j) for i in range(n) for j in range(n) if P[i, j] == 1]
    zero_rows = [i for i in range(n) if not P[i, :].any()]
    zero_cols = [j for j in range(n) if not P[:, j].any()]
    chiR = [0]*(n+1)  # chiR[k] for k=1..n
    chiC = [0]*(n+1)
    r0 = [0]*(n+1)
    c0 = [0]*(n+1)
    for k in range(1, n+1):
        # indices are 1-based in math; here i,j are 0-based, so "index <= k" means (i+1)<=k i.e. i<k
        chiR[k] = sum(1 for (i, j) in matches if (i+1) <= k < (j+1))
        chiC[k] = sum(1 for (i, j) in matches if (j+1) <= k < (i+1))
        r0[k] = sum(1 for i in zero_rows if (i+1) <= k)
        c0[k] = sum(1 for j in zero_cols if (j+1) <= k)
    return chiR, chiC, r0, c0

def check_nullity_formulas(P):
    n = P.shape[0]
    chiR, chiC, r0, c0 = combinatorial_quantities(P)
    ok = True
    for k in range(1, n+1):
        Pk   = P[:k, :k]
        Pcol = P[:, :k]      # n x k
        Prow = P[:k, :]      # k x n
        nul_Pk   = Pk.shape[1] - rank(Pk)            # square: left=right nullity dim
        nul_Pcol = Pcol.shape[1] - rank(Pcol)        # right nullity of n x k
        nul_ProwT = Prow.shape[0] - rank(Prow)       # right nullity of (k x n)^T = k - rank
        # claimed formulas
        if nul_Pk != r0[k] + chiR[k]:
            ok = False
        if nul_Pk != c0[k] + chiC[k]:
            ok = False
        if nul_Pcol != c0[k]:
            ok = False
        if nul_ProwT != r0[k]:
            ok = False
    return ok

def condition2_holds(A):
    n = A.shape[0]
    for k in range(1, n+1):
        Ak   = A[:k, :k]
        Acol = A[:, :k]
        Arow = A[:k, :]
        nul_Ak    = Ak.shape[1] - rank(Ak)
        nul_Acol  = Acol.shape[1] - rank(Acol)
        nul_ArowT = Arow.shape[0] - rank(Arow)
        if nul_Ak > nul_Acol + nul_ArowT:
            return False
    return True

def combinatorial_condition_holds(P):
    n = P.shape[0]
    chiR, chiC, r0, c0 = combinatorial_quantities(P)
    for k in range(1, n+1):
        if not (chiR[k] <= c0[k] and chiC[k] <= r0[k]):
            return False
    return True

# ---- Test A: the three nullity formulas on many random partial perms ----
fail_formulas = 0
N_TESTS = 4000
for _ in range(N_TESTS):
    n = int(rng.integers(1, 8))
    P = random_partial_perm(n, density=float(rng.uniform(0.3, 1.0)))
    if not check_nullity_formulas(P):
        fail_formulas += 1
print(f"[A] nullity formulas on {N_TESTS} random partial perms: failures = {fail_formulas}")

# ---- Test B: equivalence of condition (2) with combinatorial condition ----
mismatch = 0
agree = 0
for _ in range(N_TESTS):
    n = int(rng.integers(1, 8))
    P = random_partial_perm(n, density=float(rng.uniform(0.3, 1.0)))
    a = condition2_holds(P)
    b = combinatorial_condition_holds(P)
    if a != b:
        mismatch += 1
    else:
        agree += 1
print(f"[B] equivalence (2) <=> combinatorial on {N_TESTS} random partial perms: mismatches = {mismatch}, agreements = {agree}")

# ---- Test B2: exhaustive over ALL partial perms for small n ----
def all_partial_perms(n):
    # all injective partial maps col -> row, including the empty assignments
    idxs = range(n)
    out = []
    # choose for each column either None or a row, injective
    def rec(col, used, P):
        if col == n:
            out.append(P.copy())
            return
        # option: column col unmatched
        rec(col+1, used, P)
        for i in range(n):
            if i not in used:
                P[i, col] = 1
                rec(col+1, used | {i}, P)
                P[i, col] = 0
    rec(0, set(), np.zeros((n, n), dtype=int))
    return out

exhaustive_fail_formulas = 0
exhaustive_mismatch = 0
total = 0
for n in range(1, 5):
    for P in all_partial_perms(n):
        total += 1
        if not check_nullity_formulas(P):
            exhaustive_fail_formulas += 1
        if condition2_holds(P) != combinatorial_condition_holds(P):
            exhaustive_mismatch += 1
print(f"[B2] exhaustive n=1..4: {total} partial perms; formula failures = {exhaustive_fail_formulas}, equivalence mismatches = {exhaustive_mismatch}")

# ---- Test C: condition transfers under B = M A N (Step 2) ----
def random_invertible_lower(n):
    while True:
        L = np.tril(rng.integers(-2, 3, size=(n, n)))
        if np.all(np.diag(L) != 0):
            return L.astype(float)

def random_invertible_upper(n):
    while True:
        U = np.triu(rng.integers(-2, 3, size=(n, n)))
        if np.all(np.diag(U) != 0):
            return U.astype(float)

transfer_fail = 0
transfer_tests = 2000
for _ in range(transfer_tests):
    n = int(rng.integers(1, 7))
    A = rng.integers(-2, 3, size=(n, n)).astype(float)
    M = random_invertible_lower(n)
    Nm = random_invertible_upper(n)
    B = M @ A @ Nm
    # block identities
    bad = False
    for k in range(1, n+1):
        lhs1 = B[:k, :k]
        rhs1 = M[:k, :k] @ A[:k, :k] @ Nm[:k, :k]
        lhs2 = B[:, :k]
        rhs2 = M @ A[:, :k] @ Nm[:k, :k]
        lhs3 = B[:k, :]
        rhs3 = M[:k, :k] @ A[:k, :] @ Nm
        if not (np.allclose(lhs1, rhs1) and np.allclose(lhs2, rhs2) and np.allclose(lhs3, rhs3)):
            bad = True
    # nullity invariance
    if condition2_holds(A) != condition2_holds(B):
        bad = True
    if bad:
        transfer_fail += 1
print(f"[C] Step-2 block identities + nullity invariance on {transfer_tests} tests: failures = {transfer_fail}")

# ---- Test D: specific cases ----
swap = np.array([[0, 1], [1, 0]])
print(f"[D] swap [[0,1],[1,0]]: condition2 = {condition2_holds(swap)} (expect False), "
      f"combinatorial = {combinatorial_condition_holds(swap)} (expect False)")

# some partial perms with zero rows/cols that DO satisfy (2)
examples = {
    "zero matrix 3x3": np.zeros((3, 3), dtype=int),
    "identity 3": np.eye(3, dtype=int),
    "above-arc rooted (1->3), zero col1": np.array([[0,0,1],[0,0,0],[0,0,0]]),
    "below-arc (3->1), zero row1": np.array([[0,0,0],[0,0,0],[1,0,0]]),
    "thread: zerocol@1, arc 2->3 ... no": np.array([[0,0,0],[0,0,1],[0,0,0]]),
}
for name, P in examples.items():
    c2 = condition2_holds(P)
    cc = combinatorial_condition_holds(P)
    print(f"[D] {name}: condition2 = {c2}, combinatorial = {cc}, formulas_ok = {check_nullity_formulas(P)}")
