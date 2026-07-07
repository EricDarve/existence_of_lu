"""
ADVERSARIAL independent verification of proposed Step 1:
   Every A in F^{n x n} = M Pi N with M invertible lower-tri, N invertible upper-tri,
   Pi a partial permutation matrix.

I implement the construction EXACTLY as described in the proposed proof (Stages 1-3),
over GF(p) and over Q (Fraction).  I do NOT reuse the proposer's script.

Strategy:
 - Maintain working matrix W, and M, N such that A = M W N at all times (invariant).
   Equivalently W = M^{-1} A N^{-1}.
 - Start M = I, N = I, W = A.  Apply column ops (right-mult by upper-tri) and row ops
   (left-mult by lower-tri) to W, updating M, N to preserve A = M W N.
 - We follow the inductive bordering exactly: process k = 1..n; at stage k the leading
   (k-1) block of W is already a partial permutation Pi_{k-1}; we reduce the k-th
   border.
 - The "reduced coordinates" in the proof (W = M^{-1} A N^{-1}) are handled automatically
   by working with W directly and recording ops into M, N.  This is faithful because at
   the start of stage k, W[1:k-1,1:k-1] = Pi_{k-1} already (a partial permutation), which
   is precisely the inductive hypothesis.

Key check: I must verify the CLAIM that after Stages 1-2 the residual new column e is
supported only on zero rows of Pi_{k-1}, and the residual new row g only on zero columns.
And that Stage 3 produces a partial permutation.  I assert all of these AT RUNTIME, and
also assert the global invariant A = M W N continuously, and finally that W is a partial
permutation, M invertible lower-tri, N invertible upper-tri.
"""

import random
from fractions import Fraction


# ---------- field abstraction ----------
class QField:
    name = "Q"
    def __init__(self):
        pass
    def zero(self): return Fraction(0)
    def one(self): return Fraction(1)
    def add(self, a, b): return a + b
    def sub(self, a, b): return a - b
    def mul(self, a, b): return a * b
    def inv(self, a):
        if a == 0: raise ZeroDivisionError
        return Fraction(1) / a
    def is_zero(self, a): return a == 0
    def rand(self):
        return Fraction(random.randint(-3, 3))
    def conv(self, x): return Fraction(x)


class GFp:
    def __init__(self, p):
        self.p = p
        self.name = f"GF({p})"
    def zero(self): return 0
    def one(self): return 1 % self.p
    def add(self, a, b): return (a + b) % self.p
    def sub(self, a, b): return (a - b) % self.p
    def mul(self, a, b): return (a * b) % self.p
    def inv(self, a):
        a %= self.p
        if a == 0: raise ZeroDivisionError
        return pow(a, self.p - 2, self.p)
    def is_zero(self, a): return a % self.p == 0
    def rand(self):
        return random.randint(0, self.p - 1)
    def conv(self, x): return x % self.p


# ---------- dense matrix over field, as list of lists ----------
def eye(F, n):
    return [[F.one() if i == j else F.zero() for j in range(n)] for i in range(n)]

def matmul(F, A, B):
    n = len(A); m = len(B[0]); kk = len(B)
    C = [[F.zero() for _ in range(m)] for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for t in range(kk):
            a = Ai[t]
            if F.is_zero(a): continue
            Bt = B[t]
            Ci = C[i]
            for j in range(m):
                Ci[j] = F.add(Ci[j], F.mul(a, Bt[j]))
    return C

def mat_eq(F, A, B):
    if len(A) != len(B): return False
    for i in range(len(A)):
        for j in range(len(A[0])):
            if not F.is_zero(F.sub(A[i][j], B[i][j])):
                return False
    return True


# ---------- elementary ops on (W, M, N) preserving A = M W N ----------
# A = M W N.  We modify W by E_left^{-1} * W * E_right^{-1}, and set
#   M <- M E_left, N <- E_right N, so that M W N is unchanged.
# Operating on W as W <- E_left^{-1} W E_right^{-1}.

def row_op_addmul(F, W, M, src, dst, lam):
    """W: row_dst <- row_dst + lam * row_src  (a left mult by L = I + lam e_dst e_src^T).
       To keep W' = L^{-1}? -- careful: we DEFINE the new working matrix as the result of
       the row op applied to W. We must record M <- M * L^{-1}... let's instead keep
       invariant A = M W N where the row op transforms W directly: W_new = L_applied(W).
       The op 'add lam*row_src to row_dst' is left-multiplication by L=I+lam e_dst e_src^T.
       So W_new = L W. To keep A=M W N = M_new W_new N, need M_new = M L^{-1}.
       L^{-1} = I - lam e_dst e_src^T. So M_new = M (I - lam e_dst e_src^T):
         column src of M_new = col_src(M) - lam * col_dst(M)."""
    n = len(W)
    # W_new = L W : row_dst += lam*row_src
    for j in range(len(W[0])):
        W[dst][j] = F.add(W[dst][j], F.mul(lam, W[src][j]))
    # M_new = M (I - lam e_dst e_src^T): col_src(M) -= lam * col_dst(M)
    for i in range(n):
        M[i][src] = F.sub(M[i][src], F.mul(lam, M[i][dst]))

def col_op_addmul(F, W, N, src, dst, lam):
    """W: col_dst <- col_dst + lam * col_src  (right mult by U = I + lam e_src e_dst^T).
       W_new = W U. To keep A = M W N: M W_new N_new = M W U N_new = A = M W N
       => U N_new = N => N_new = U^{-1} N = (I - lam e_src e_dst^T) N:
         row_src(N) -= lam * row_dst(N)."""
    n = len(W)
    for i in range(len(W)):
        W[i][dst] = F.add(W[i][dst], F.mul(lam, W[i][src]))
    for j in range(n):
        N[src][j] = F.sub(N[src][j], F.mul(lam, N[dst][j]))

def scale_row(F, W, M, idx, c):
    """W: row_idx <- c * row_idx (left mult by S=diag with c at idx).
       W_new = S W. M_new = M S^{-1}: col_idx(M) *= c^{-1}."""
    cinv = F.inv(c)
    for j in range(len(W[0])):
        W[idx][j] = F.mul(c, W[idx][j])
    for i in range(len(W)):
        M[i][idx] = F.mul(M[i][idx], cinv)

def scale_col(F, W, N, idx, c):
    """W: col_idx <- c*col_idx (right mult by S). W_new = W S. N_new = S^{-1} N:
        row_idx(N) *= c^{-1}."""
    cinv = F.inv(c)
    for i in range(len(W)):
        W[i][idx] = F.mul(c, W[i][idx])
    for j in range(len(W)):
        N[idx][j] = F.mul(N[idx][j], cinv)


def is_lower_tri_invertible(F, M):
    n = len(M)
    for i in range(n):
        for j in range(i+1, n):
            if not F.is_zero(M[i][j]): return False
        if F.is_zero(M[i][i]): return False
    return True

def is_upper_tri_invertible(F, N):
    n = len(N)
    for i in range(n):
        for j in range(i):
            if not F.is_zero(N[i][j]): return False
        if F.is_zero(N[i][i]): return False
    return True

def is_partial_perm(F, W):
    n = len(W)
    for i in range(n):
        cnt = 0
        for j in range(n):
            v = W[i][j]
            if F.is_zero(v): continue
            if not F.is_zero(F.sub(v, F.one())): return False  # not 0/1
            cnt += 1
        if cnt > 1: return False
    for j in range(n):
        cnt = 0
        for i in range(n):
            if not F.is_zero(W[i][j]): cnt += 1
        if cnt > 1: return False
    return True


def reduce_to_partial_perm(F, A, verbose=False):
    """Implement proposed Stages 1-3 inductively. Returns (M, Pi, N).
       Asserts invariant A = M W N throughout and structural claims at each stage."""
    n = len(A)
    W = [row[:] for row in A]
    M = eye(F, n)
    N = eye(F, n)
    A0 = [row[:] for row in A]

    def check_invariant(tag):
        prod = matmul(F, matmul(F, M, W), N)
        if not mat_eq(F, prod, A0):
            raise AssertionError(f"INVARIANT A=MWN BROKEN at {tag}")

    # process columns/rows k = 0..n-1 (0-indexed). At start of stage k, W[0:k,0:k] is
    # a partial permutation (inductive hypothesis); we verify it.
    for k in range(n):
        # --- determine matched pairs and zero rows/cols within leading k x k block ---
        # leading block = indices 0..k-1
        # For Pi_{k-1}: which row i<k has a 1, at which column j<k.
        row_match = {}   # i -> j  (matched pairs in leading block)
        col_match = {}   # j -> i
        if k > 0:
            for i in range(k):
                jset = [j for j in range(k) if not F.is_zero(W[i][j])]
                # inductive hypothesis: at most one, and equal to 1
                assert len(jset) <= 1, f"k={k}: leading block row {i} not partial perm"
                if jset:
                    j = jset[0]
                    assert F.is_zero(F.sub(W[i][j], F.one())), "leading entry not 1"
                    row_match[i] = j
                    col_match[j] = i
            # also verify column side
            for j in range(k):
                iset = [i for i in range(k) if not F.is_zero(W[i][j])]
                assert len(iset) <= 1, f"k={k}: leading block col {j} not partial perm"

        zero_rows = [i for i in range(k) if i not in row_match]
        zero_cols = [j for j in range(k) if j not in col_match]

        # === Stage 1: clear new column (column k) over matched rows ===
        # For each matched pair (i,j): subtract W[i][k] times column j from column k.
        for (i, j) in list(row_match.items()):
            lam = W[i][k]
            if not F.is_zero(lam):
                col_op_addmul(F, W, N, src=j, dst=k, lam=F.sub(F.zero(), lam))
        # CLAIM: residual new column e = W[0:k, k] supported only on zero rows.
        for i in range(k):
            if i in row_match:
                assert F.is_zero(W[i][k]), (
                    f"STAGE1 CLAIM FAIL k={k}: W[{i}][{k}] not cleared in matched row")

        # === Stage 2: clear new row (row k) over matched columns ===
        for (j, i) in list(col_match.items()):
            lam = W[k][j]
            if not F.is_zero(lam):
                row_op_addmul(F, W, M, src=i, dst=k, lam=F.sub(F.zero(), lam))
        for j in range(k):
            if j in col_match:
                assert F.is_zero(W[k][j]), (
                    f"STAGE2 CLAIM FAIL k={k}: W[{k}][{j}] not cleared in matched col")

        check_invariant(f"after stage2 k={k}")

        # === Stage 3: normalize residual column, residual row, corner ===
        # -- Column normalization --
        col_match_created = None
        e_nonzero = [i for i in zero_rows if not F.is_zero(W[i][k])]
        if e_nonzero:
            istar = e_nonzero[0]
            # scale column k by W[istar][k]^{-1} so pivot becomes 1
            scale_col(F, W, N, k, F.inv(W[istar][k]))
            assert F.is_zero(F.sub(W[istar][k], F.one())), "istar not normalized to 1"
            # clear later zero-row entries istar < i < k
            for i in zero_rows:
                if i > istar and not F.is_zero(W[i][k]):
                    # row_i <- row_i - W[i][k]*row_istar  (lower tri since istar < i)
                    row_op_addmul(F, W, M, src=istar, dst=i, lam=F.sub(F.zero(), W[i][k]))
            # verify: residual column now single 1 at (istar,k)
            for i in range(k):
                if i == istar:
                    assert F.is_zero(F.sub(W[i][k], F.one()))
                else:
                    assert F.is_zero(W[i][k]), f"col not single-1 at row {i}"
            col_match_created = istar
            assert istar < k, "istar must be < k for lower-tri op"

        # -- Row normalization --
        row_match_created = None
        g_nonzero = [j for j in zero_cols if not F.is_zero(W[k][j])]
        if g_nonzero:
            jstar = g_nonzero[0]
            scale_row(F, W, M, k, F.inv(W[k][jstar]))
            assert F.is_zero(F.sub(W[k][jstar], F.one()))
            for j in zero_cols:
                if j > jstar and not F.is_zero(W[k][j]):
                    # col_j <- col_j - W[k][j]*col_jstar (upper tri since jstar < j)
                    col_op_addmul(F, W, N, src=jstar, dst=j, lam=F.sub(F.zero(), W[k][j]))
            for j in range(k):
                if j == jstar:
                    assert F.is_zero(F.sub(W[k][j], F.one()))
                else:
                    assert F.is_zero(W[k][j]), f"row not single-1 at col {j}"
            row_match_created = jstar
            assert jstar < k

        # -- Corner reconciliation --
        w = W[k][k]
        if col_match_created is not None:
            istar = col_match_created
            # row_k <- row_k - w*row_istar (row istar has support only {(istar,k)})
            if not F.is_zero(w):
                row_op_addmul(F, W, M, src=istar, dst=k, lam=F.sub(F.zero(), w))
            assert F.is_zero(W[k][k]), f"corner not cleared (col match) k={k}"
        elif row_match_created is not None:
            jstar = row_match_created
            if not F.is_zero(w):
                col_op_addmul(F, W, N, src=jstar, dst=k, lam=F.sub(F.zero(), w))
            assert F.is_zero(W[k][k]), f"corner not cleared (row match) k={k}"
        else:
            # both e=0 and g=0: block is Pi_{k-1} + [w]
            if not F.is_zero(w):
                scale_col(F, W, N, k, F.inv(w))
                assert F.is_zero(F.sub(W[k][k], F.one()))
            # else: isolated zero row & col k

        check_invariant(f"after stage3 k={k}")

        # verify leading (k+1) block is a partial permutation
        for i in range(k+1):
            cnt = sum(1 for j in range(k+1) if not F.is_zero(W[i][j]))
            assert cnt <= 1, f"k={k}: row {i} has {cnt} nonzeros in leading block"
        for j in range(k+1):
            cnt = sum(1 for i in range(k+1) if not F.is_zero(W[i][j]))
            assert cnt <= 1, f"k={k}: col {j} has {cnt} nonzeros in leading block"

    # final checks
    check_invariant("final")
    assert is_partial_perm(F, W), "final W not a partial permutation"
    assert is_lower_tri_invertible(F, M), "M not invertible lower-tri"
    assert is_upper_tri_invertible(F, N), "N not invertible upper-tri"
    return M, W, N


def rand_matrix(F, n, density=0.5):
    A = [[F.zero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if random.random() < density:
                A[i][j] = F.rand()
    return A


def run_fuzz(F, trials, nmax=7):
    passes = 0
    for _ in range(trials):
        n = random.randint(1, nmax)
        dens = random.choice([0.2, 0.4, 0.6, 0.8, 1.0])
        A = rand_matrix(F, n, dens)
        try:
            reduce_to_partial_perm(F, A)
            passes += 1
        except Exception as ex:
            print(f"  [{F.name}] FAIL n={n} dens={dens}: {ex}")
            for r in A: print("   ", r)
            return passes, False
    return passes, True


def run_hand_cases(F):
    cases = {
        "[0]": [[0]],
        "swap [[0,1],[1,0]]": [[0,1],[1,0]],
        "2x2 zero": [[0,0],[0,0]],
        "[[0,0,0],[0,0,1],[0,1,0]]": [[0,0,0],[0,0,1],[0,1,0]],
        "[[1,2,3],[4,5,6],[7,8,0]]": [[1,2,3],[4,5,6],[7,8,0]],
        "[[0,0,1],[0,1,0],[1,0,0]]": [[0,0,1],[0,1,0],[1,0,0]],
        "3x3 all ones": [[1,1,1],[1,1,1],[1,1,1]],
        "[[0,1,0],[1,0,0],[0,0,0]]": [[0,1,0],[1,0,0],[0,0,0]],
        "single col nonzero": [[0,0,0],[1,0,0],[0,0,0]],
    }
    ok = True
    for name, A in cases.items():
        Af = [[F.conv(x) for x in row] for row in A]
        try:
            reduce_to_partial_perm(F, Af)
        except Exception as ex:
            print(f"  [{F.name}] HAND FAIL {name}: {ex}")
            ok = False
    return ok


def main():
    random.seed(20260607)
    fields = [QField(), GFp(2), GFp(3), GFp(5), GFp(7)]
    total_fail = 0
    for F in fields:
        p, ok = run_fuzz(F, 600, nmax=7)
        hok = run_hand_cases(F)
        status = "OK" if (ok and hok) else "FAIL"
        print(f"[{F.name:8s}] fuzz_passes={p}  hand={'OK' if hok else 'FAIL'}  => {status}")
        if not (ok and hok): total_fail += 1
    # larger
    for F in [QField(), GFp(2), GFp(3)]:
        passes = 0
        ok_all = True
        for _ in range(120):
            n = random.randint(8, 12)
            A = rand_matrix(F, n, random.choice([0.3, 0.6, 1.0]))
            try:
                reduce_to_partial_perm(F, A)
                passes += 1
            except Exception as ex:
                print(f"  LARGE [{F.name}] FAIL n={n}: {ex}")
                ok_all = False
                break
        print(f"large [{F.name:6s}] passes={passes} {'OK' if ok_all else 'FAIL'}")
        if not ok_all: total_fail += 1
    print("TOTAL FIELD-FAILS:", total_fail)


if __name__ == "__main__":
    main()
