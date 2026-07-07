"""
Independent adversarial verification of the PROPOSED Step 2 + combinatorial
restatement for the UNSYMMETRIC LU criterion.

Key independence choices vs. the author's script:
 - EXACT rank over the rationals via Fraction-based Gaussian elimination
   (no floating point np.linalg.matrix_rank).
 - Exact rank over GF(2) as a separate independent check (char 2 matters for
   field-generality claims).
 - Independent re-implementation of every combinatorial quantity, with a
   deliberately different counting convention (0-based but re-derived).
 - Exhaustive enumeration of ALL partial permutations up to n=5.
 - Step-2 transfer tested over Q with exact Fraction matrices AND over GF(2).

We verify:
  (F1) nullity(Pi[:k,:k]) = r0_k + chiR_k = c0_k + chiC_k
  (F2) nullity(Pi[:,:k]) = c0_k
  (F3) nullity(Pi[:k,:]^T) = r0_k
  bookkeeping (3.1)/(3.2)/(3.3)
  equivalence: (2) <=> (chiR_k<=c0_k and chiC_k<=r0_k for all k)
  Step 2 block identities (2.3)-(2.5) and nullity invariance under B = M A N
"""

from fractions import Fraction
import itertools
import random

random.seed(12345)

# ----------------------------------------------------------------------
# Exact rank over Q (Fraction Gaussian elimination)
# ----------------------------------------------------------------------
def rank_Q(M):
    # M: list of lists of Fraction (or int)
    if len(M) == 0:
        return 0
    A = [[Fraction(x) for x in row] for row in M]
    nrows = len(A)
    ncols = len(A[0]) if nrows else 0
    rank = 0
    pivot_row = 0
    for col in range(ncols):
        # find pivot
        piv = None
        for r in range(pivot_row, nrows):
            if A[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        A[pivot_row], A[piv] = A[piv], A[pivot_row]
        pivval = A[pivot_row][col]
        for r in range(nrows):
            if r != pivot_row and A[r][col] != 0:
                factor = A[r][col] / pivval
                A[r] = [A[r][c] - factor * A[pivot_row][c] for c in range(ncols)]
        pivot_row += 1
        rank += 1
        if pivot_row == nrows:
            break
    return rank

# ----------------------------------------------------------------------
# Exact rank over GF(2)
# ----------------------------------------------------------------------
def rank_GF2(M):
    if len(M) == 0:
        return 0
    A = [[int(x) & 1 for x in row] for row in M]
    nrows = len(A)
    ncols = len(A[0]) if nrows else 0
    rank = 0
    pivot_row = 0
    for col in range(ncols):
        piv = None
        for r in range(pivot_row, nrows):
            if A[r][col] == 1:
                piv = r
                break
        if piv is None:
            continue
        A[pivot_row], A[piv] = A[piv], A[pivot_row]
        for r in range(nrows):
            if r != pivot_row and A[r][col] == 1:
                A[r] = [(A[r][c] ^ A[pivot_row][c]) for c in range(ncols)]
        pivot_row += 1
        rank += 1
        if pivot_row == nrows:
            break
    return rank

# ----------------------------------------------------------------------
# Matrix helpers (plain lists)
# ----------------------------------------------------------------------
def submatrix(M, rows, cols):
    return [[M[i][j] for j in cols] for i in rows]

def transpose(M):
    if not M:
        return []
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]

def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    out = [[Fraction(0)]*p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for k in range(m):
            a = Ai[k]
            if a == 0:
                continue
            Bk = B[k]
            outi = out[i]
            for j in range(p):
                outi[j] += a * Bk[j]
    return out

def matmul_gf2(A, B):
    n, m, p = len(A), len(B), len(B[0])
    out = [[0]*p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            if A[i][k] & 1:
                for j in range(p):
                    out[i][j] ^= (B[k][j] & 1)
    return out

# ----------------------------------------------------------------------
# Partial permutation generation
# ----------------------------------------------------------------------
def random_partial_perm(n, density):
    """At most one 1 per row and per column."""
    P = [[0]*n for _ in range(n)]
    rows = list(range(n))
    random.shuffle(rows)
    used_cols = set()
    for i in rows:
        if random.random() < density:
            avail = [c for c in range(n) if c not in used_cols]
            if not avail:
                continue
            j = random.choice(avail)
            P[i][j] = 1
            used_cols.add(j)
    return P

def all_partial_perms(n):
    """ALL injective partial maps row->col (incl. empty)."""
    out = []
    def rec(row, used_cols, P):
        if row == n:
            out.append([r[:] for r in P])
            return
        rec(row+1, used_cols, P)          # row unmatched
        for j in range(n):
            if j not in used_cols:
                P[row][j] = 1
                rec(row+1, used_cols | {j}, P)
                P[row][j] = 0
    rec(0, set(), [[0]*n for _ in range(n)])
    return out

# ----------------------------------------------------------------------
# Combinatorial quantities (independent, 1-based indices i,j = position+1)
# ----------------------------------------------------------------------
def combinatorial(P):
    n = len(P)
    matches = [(i, j) for i in range(n) for j in range(n) if P[i][j] == 1]
    zero_rows = [i for i in range(n) if all(P[i][j] == 0 for j in range(n))]
    zero_cols = [j for j in range(n) if all(P[i][j] == 0 for i in range(n))]
    chiR = [0]*(n+1)
    chiC = [0]*(n+1)
    r0 = [0]*(n+1)
    c0 = [0]*(n+1)
    mk = [0]*(n+1)
    for k in range(1, n+1):
        # math index of row i is (i+1); condition i+1 <= k < j+1
        chiR[k] = sum(1 for (i, j) in matches if (i+1) <= k < (j+1))
        chiC[k] = sum(1 for (i, j) in matches if (j+1) <= k < (i+1))
        r0[k]   = sum(1 for i in zero_rows if (i+1) <= k)
        c0[k]   = sum(1 for j in zero_cols if (j+1) <= k)
        mk[k]   = sum(1 for (i, j) in matches if (i+1) <= k and (j+1) <= k)
    return chiR, chiC, r0, c0, mk

# ----------------------------------------------------------------------
# Condition (2) directly, via exact rank
# ----------------------------------------------------------------------
def condition2(A, rankfn):
    n = len(A)
    for k in range(1, n+1):
        rows_k = list(range(k))
        cols_k = list(range(k))
        all_idx = list(range(n))
        Ak  = submatrix(A, rows_k, cols_k)          # k x k
        Acol = submatrix(A, all_idx, cols_k)        # n x k
        Arow = submatrix(A, rows_k, all_idx)        # k x n
        nul_Ak   = k - rankfn(Ak)
        nul_Acol = k - rankfn(Acol)                 # right nullity n x k = k - rank
        nul_ArowT = k - rankfn(Arow)                # right nullity of (k x n)^T = k - rank
        if nul_Ak > nul_Acol + nul_ArowT:
            return False
    return True

def combinatorial_condition(P):
    n = len(P)
    chiR, chiC, r0, c0, mk = combinatorial(P)
    for k in range(1, n+1):
        if not (chiR[k] <= c0[k] and chiC[k] <= r0[k]):
            return False
    return True

def check_formulas(P, rankfn):
    n = len(P)
    chiR, chiC, r0, c0, mk = combinatorial(P)
    msgs = []
    for k in range(1, n+1):
        rows_k = list(range(k)); cols_k = list(range(k)); allx = list(range(n))
        Pk  = submatrix(P, rows_k, cols_k)
        Pcol = submatrix(P, allx, cols_k)
        Prow = submatrix(P, rows_k, allx)
        nul_Pk    = k - rankfn(Pk)
        nul_Pcol  = k - rankfn(Pcol)
        nul_ProwT = k - rankfn(Prow)
        # bookkeeping
        if k != r0[k] + mk[k] + chiR[k]:
            msgs.append(f"k={k} bookkeeping(3.1) fail")
        if k != c0[k] + mk[k] + chiC[k]:
            msgs.append(f"k={k} bookkeeping(3.2) fail")
        if r0[k] + chiR[k] != c0[k] + chiC[k]:
            msgs.append(f"k={k} (3.3) fail")
        # F1
        if nul_Pk != r0[k] + chiR[k]:
            msgs.append(f"k={k} F1a fail: nul_Pk={nul_Pk} r0+chiR={r0[k]+chiR[k]}")
        if nul_Pk != c0[k] + chiC[k]:
            msgs.append(f"k={k} F1b fail: nul_Pk={nul_Pk} c0+chiC={c0[k]+chiC[k]}")
        # F2
        if nul_Pcol != c0[k]:
            msgs.append(f"k={k} F2 fail: nul_Pcol={nul_Pcol} c0={c0[k]}")
        # F3
        if nul_ProwT != r0[k]:
            msgs.append(f"k={k} F3 fail: nul_ProwT={nul_ProwT} r0={r0[k]}")
    return msgs

# ======================================================================
# TEST A: formulas + bookkeeping on random partial perms (over Q)
# ======================================================================
failA = 0
NA = 5000
for _ in range(NA):
    n = random.randint(1, 8)
    P = random_partial_perm(n, random.uniform(0.2, 1.0))
    msgs = check_formulas(P, rank_Q)
    if msgs:
        failA += 1
        if failA <= 3:
            print("  [A] FAIL", P, msgs)
print(f"[A] formulas+bookkeeping over Q on {NA} random partial perms: failures = {failA}")

# ======================================================================
# TEST A': same formulas over GF(2)
# ======================================================================
failAp = 0
for _ in range(NA):
    n = random.randint(1, 8)
    P = random_partial_perm(n, random.uniform(0.2, 1.0))
    msgs = check_formulas(P, rank_GF2)
    if msgs:
        failAp += 1
        if failAp <= 3:
            print("  [A'] FAIL", P, msgs)
print(f"[A'] formulas+bookkeeping over GF(2) on {NA} random partial perms: failures = {failAp}")

# ======================================================================
# TEST B: equivalence (2) <=> combinatorial, over Q and GF(2)
# ======================================================================
def equiv_test(rankfn, label):
    mismatch = 0; agree = 0
    for _ in range(NA):
        n = random.randint(1, 8)
        P = random_partial_perm(n, random.uniform(0.2, 1.0))
        if condition2(P, rankfn) != combinatorial_condition(P):
            mismatch += 1
        else:
            agree += 1
    print(f"[B/{label}] (2)<=>combinatorial on {NA}: mismatches={mismatch} agreements={agree}")
equiv_test(rank_Q, "Q")
equiv_test(rank_GF2, "GF2")

# ======================================================================
# TEST B2: exhaustive over ALL partial perms up to n=5
# ======================================================================
ex_form_fail = 0
ex_mismatch = 0
ex_total = 0
for n in range(1, 6):
    for P in all_partial_perms(n):
        ex_total += 1
        if check_formulas(P, rank_Q):
            ex_form_fail += 1
        if condition2(P, rank_Q) != combinatorial_condition(P):
            ex_mismatch += 1
print(f"[B2] EXHAUSTIVE n=1..5: {ex_total} partial perms; "
      f"formula failures={ex_form_fail}, equivalence mismatches={ex_mismatch}")

# ======================================================================
# TEST C: Step-2 block identities + nullity invariance, over Q (exact Fractions)
# ======================================================================
def rand_inv_lower_Q(n):
    while True:
        L = [[Fraction(random.randint(-2, 2)) if j <= i else Fraction(0)
              for j in range(n)] for i in range(n)]
        if all(L[i][i] != 0 for i in range(n)):
            return L

def rand_inv_upper_Q(n):
    while True:
        U = [[Fraction(random.randint(-2, 2)) if j >= i else Fraction(0)
              for j in range(n)] for i in range(n)]
        if all(U[i][i] != 0 for i in range(n)):
            return U

def rank_Q_frac(M):
    return rank_Q(M)

failC = 0
NC = 3000
for _ in range(NC):
    n = random.randint(1, 6)
    A = [[Fraction(random.randint(-2, 2)) for _ in range(n)] for _ in range(n)]
    M = rand_inv_lower_Q(n)
    Nm = rand_inv_upper_Q(n)
    B = matmul(matmul(M, A), Nm)
    bad = False
    for k in range(1, n+1):
        rk = list(range(k)); ck = list(range(k)); allx = list(range(n))
        # (2.3)
        lhs = submatrix(B, rk, ck)
        rhs = matmul(matmul(submatrix(M, rk, ck), submatrix(A, rk, ck)), submatrix(Nm, ck, ck))
        if lhs != rhs:
            bad = True; break
        # (2.4)
        lhs = submatrix(B, allx, ck)
        rhs = matmul(matmul(M, submatrix(A, allx, ck)), submatrix(Nm, ck, ck))
        if lhs != rhs:
            bad = True; break
        # (2.5)
        lhs = submatrix(B, rk, allx)
        rhs = matmul(matmul(submatrix(M, rk, ck), submatrix(A, rk, allx)), Nm)
        if lhs != rhs:
            bad = True; break
    if not bad:
        if condition2(A, rank_Q) != condition2(B, rank_Q):
            bad = True
    if bad:
        failC += 1
        if failC <= 3:
            print("  [C] FAIL n=", n)
print(f"[C] Step-2 block identities + nullity invariance over Q on {NC}: failures = {failC}")

# Also test Step 2 block identities + invariance over GF(2)
def rand_inv_lower_gf2(n):
    L = [[ (random.randint(0,1) if j < i else (1 if j==i else 0)) for j in range(n)] for i in range(n)]
    return L
def rand_inv_upper_gf2(n):
    U = [[ (random.randint(0,1) if j > i else (1 if j==i else 0)) for j in range(n)] for i in range(n)]
    return U

failC2 = 0
NC2 = 2000
for _ in range(NC2):
    n = random.randint(1, 6)
    A = [[random.randint(0,1) for _ in range(n)] for _ in range(n)]
    M = rand_inv_lower_gf2(n)
    Nm = rand_inv_upper_gf2(n)
    B = matmul_gf2(matmul_gf2(M, A), Nm)
    if condition2(A, rank_GF2) != condition2(B, rank_GF2):
        failC2 += 1
print(f"[C/GF2] nullity invariance under B=MAN over GF(2) on {NC2}: failures = {failC2}")

# ======================================================================
# TEST D: the forbidden swap and assorted cases
# ======================================================================
swap = [[0,1],[1,0]]
print(f"[D] swap [[0,1],[1,0]]: cond2(Q)={condition2(swap, rank_Q)} "
      f"comb={combinatorial_condition(swap)} (expect False,False)")
chiR, chiC, r0, c0, mk = combinatorial(swap)
print(f"     swap details: chiR={chiR[1:]} c0={c0[1:]} chiC={chiC[1:]} r0={r0[1:]}")

cases = {
    "zero 3x3": [[0,0,0],[0,0,0],[0,0,0]],
    "identity 3": [[1,0,0],[0,1,0],[0,0,1]],
    "above-arc (1->3) zerocol1": [[0,0,1],[0,0,0],[0,0,0]],
    "below-arc (3->1) zerorow1": [[0,0,0],[0,0,0],[1,0,0]],
    "swap-like 3 (2<->3)": [[0,0,0],[0,0,1],[0,1,0]],
    "full reversal n=3": [[0,0,1],[0,1,0],[1,0,0]],
}
for name, P in cases.items():
    c2 = condition2(P, rank_Q)
    cc = combinatorial_condition(P)
    fm = (check_formulas(P, rank_Q) == [])
    print(f"[D] {name}: cond2={c2} comb={cc} formulas_ok={fm}")
