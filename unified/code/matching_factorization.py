# scifig_start
"""Constructive unpivoted LU and LDL^T factorizations via matching normal forms.

All arithmetic is exact and field-generic: matrices are lists of lists of
field elements (fractions.Fraction for Q, Fp for GF(p)).  A `field` object
provides the constants `zero` and `one`.  Zeros are allowed on the diagonals
of all triangular factors; the input matrix may be singular.

The two entry points are
    lu_factorization(A, field)    -> (L, U)    or None,
    ldlt_factorization(A, field)  -> (L, d)    or None,
and each returns None exactly when the corresponding nullity criterion
fails (Theorems A and B of the paper).
"""

from fractions import Fraction
import bisect


# ---------------------------------------------------------------- fields
class QField:
    """The rationals."""
    zero = Fraction(0)
    one = Fraction(1)

    @staticmethod
    def of(n):
        return Fraction(n)


class Fp:
    """An element of the prime field GF(p)."""
    __slots__ = ('p', 'v')

    def __init__(self, p, v):
        self.p = p
        self.v = v % p

    def __add__(self, o):
        return Fp(self.p, self.v + o.v)

    def __sub__(self, o):
        return Fp(self.p, self.v - o.v)

    def __neg__(self):
        return Fp(self.p, -self.v)

    def __mul__(self, o):
        return Fp(self.p, self.v * o.v)

    def __truediv__(self, o):
        assert o.v != 0
        return Fp(self.p, self.v * pow(o.v, self.p - 2, self.p))

    def __eq__(self, o):
        return isinstance(o, Fp) and self.p == o.p and self.v == o.v

    def __hash__(self):
        return hash((self.p, self.v))

    def __repr__(self):
        return str(self.v)


class GFField:
    """The prime field GF(p)."""
    def __init__(self, p):
        self.p = p
        self.zero = Fp(p, 0)
        self.one = Fp(p, 1)

    def of(self, n):
        return Fp(self.p, n)


# ------------------------------------------------------- small linear algebra
def zeros(m, n, field):
    return [[field.zero] * n for _ in range(m)]


def eye(n, field):
    Id = zeros(n, n, field)
    for i in range(n):
        Id[i][i] = field.one
    return Id


def mat_mul(A, B, field):
    m, k, n = len(A), len(B), len(B[0])
    C = zeros(m, n, field)
    for i in range(m):
        Ai = A[i]
        for t in range(k):
            a = Ai[t]
            if a != field.zero:
                Bt = B[t]
                Ci = C[i]
                for j in range(n):
                    Ci[j] = Ci[j] + a * Bt[j]
    return C


def transpose(A):
    return [list(col) for col in zip(*A)]


def mat_eq(A, B):
    return A == B


def is_lower(A, field):
    return all(A[i][j] == field.zero for i in range(len(A))
               for j in range(i + 1, len(A)))


def is_upper(A, field):
    return all(A[i][j] == field.zero for i in range(len(A))
               for j in range(i))


def rank(A, field):
    """Rank by Gaussian elimination with unrestricted (full) pivoting.

    This routine is used only to *evaluate* the criteria; the factorization
    algorithms below never permute anything.
    """
    if not A or not A[0]:
        return 0
    A = [row[:] for row in A]
    m, n = len(A), len(A[0])
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, m) if A[i][c] != field.zero), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = A[r][c]
        for i in range(r + 1, m):
            if A[i][c] != field.zero:
                t = A[i][c] / inv
                A[i] = [A[i][j] - t * A[r][j] for j in range(n)]
        r += 1
        if r == m:
            break
    return r


def forward_solve(M, b, field):
    """Solve M x = b with M lower triangular, nonzero diagonal."""
    k = len(M)
    x = [field.zero] * k
    for i in range(k):
        s = b[i]
        for j in range(i):
            if M[i][j] != field.zero:
                s = s - M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def forward_solve_T(N, c, field):
    """Solve x^T N = c^T (i.e. N^T x = c) with N upper triangular."""
    return forward_solve(transpose(N), c, field)


# ------------------------------------------------------------ the criteria
def nullity_criterion_lu(A, field):
    """null(A[:k,:k]) <= null(A[:,:k]) + null(A[:k,:]^T) for all k."""
    n = len(A)
    for k in range(1, n + 1):
        lead = [row[:k] for row in A[:k]]
        cols = [row[:k] for row in A]
        rows = A[:k]
        if k - rank(lead, field) > (k - rank(cols, field)) + (k - rank(rows, field)):
            return False
    return True


def nullity_criterion_ldlt(A, field):
    """null(A[:k,:k]) <= 2 null(A[:,:k]) for all k (A symmetric)."""
    n = len(A)
    for k in range(1, n + 1):
        lead = [row[:k] for row in A[:k]]
        cols = [row[:k] for row in A]
        if k - rank(lead, field) > 2 * (k - rank(cols, field)):
            return False
    return True


# ------------------------------------ Step 1a: reduction A = M Pi N (Lemma)
def reduce_to_matching(A, field):
    """Return (M, pairs, N) with A = M Pi N, M invertible lower triangular,
    N invertible upper triangular, and Pi the weighted matching matrix
    Pi = sum of w * e_i e_j^T over pairs[(i, j)] = w (at most one nonzero
    per row and per column)."""
    n = len(A)
    M = [[field.one]]
    N = [[field.one]]
    pairs = {}
    if A[0][0] != field.zero:
        pairs[(0, 0)] = A[0][0]
    for k in range(1, n):
        a = [A[i][k] for i in range(k)]
        c = [A[k][j] for j in range(k)]
        b = A[k][k]
        ah = forward_solve(M, a, field)         # M^{-1} a
        ch = forward_solve_T(N, c, field)       # c^T N^{-1}
        # split off the matched part of the new column and row
        x = [field.zero] * k
        y = [field.zero] * k
        for (i, j), w in pairs.items():
            x[j] = ah[i] / w
            y[i] = ch[j] / w
        e = ah[:]
        g = ch[:]
        f = b
        for (i, j), w in pairs.items():
            e[i] = field.zero                   # e = ah - Pi x
            g[j] = field.zero                   # g^T = ch^T - y^T Pi
            f = f - y[i] * w * x[j]             # f = b - y^T Pi x
        # grow the outer factors: last row of M is (y,1), last col of N is x
        for i in range(k):
            M[i].append(field.zero)
            N[i].append(x[i])
        M.append(y + [field.one])
        N.append([field.zero] * k + [field.one])
        # residual column -> new pair (istar, k)
        istar = next((i for i in range(k) if e[i] != field.zero), None)
        if istar is not None:
            alpha = e[istar]
            for i in range(istar + 1, k):       # M <- M R: col istar update
                if e[i] != field.zero:
                    t = e[i] / alpha
                    for r in range(k + 1):
                        M[r][istar] = M[r][istar] + t * M[r][i]
            pairs[(istar, k)] = alpha
        # residual row -> new pair (k, jstar)
        jstar = next((j for j in range(k) if g[j] != field.zero), None)
        if jstar is not None:
            gamma = g[jstar]
            for j in range(jstar + 1, k):       # N <- S N: row jstar update
                if g[j] != field.zero:
                    t = g[j] / gamma
                    for s in range(k + 1):
                        N[jstar][s] = N[jstar][s] + t * N[j][s]
            pairs[(k, jstar)] = gamma
        # corner
        if istar is not None:
            M[k][istar] = M[k][istar] + f / alpha
        elif jstar is not None:
            N[jstar][k] = N[jstar][k] + f / gamma
        elif f != field.zero:
            pairs[(k, k)] = f
    return M, pairs, N


# --------------------- Step 3a: LU of a matching matrix (greedy injection)
def lu_of_matching(pairs, n, field):
    """Given the weighted matching matrix Pi (as pairs), build L, U with
    L U = Pi, using the largest-free-slot greedy.  Returns None iff the
    prefix condition (equivalently, the nullity criterion for Pi) fails."""
    L = zeros(n, n, field)
    U = zeros(n, n, field)
    free = list(range(n))                       # sorted free slots
    for (i, j), w in sorted(pairs.items(), key=lambda kv: min(kv[0])):
        m = min(i, j)
        pos = bisect.bisect_right(free, m) - 1
        if pos < 0:
            return None                         # criterion fails
        t = free.pop(pos)                       # largest free slot <= m
        L[i][t] = field.one
        U[t][j] = w
    return L, U


def lu_factorization(A, field):
    """A = L U with L lower and U upper triangular (zeros allowed on both
    diagonals), or None when no such factorization exists."""
    n = len(A)
    if n == 0:
        return [], []
    M, pairs, N = reduce_to_matching(A, field)
    res = lu_of_matching(pairs, n, field)
    if res is None:
        return None
    Lp, Up = res
    return mat_mul(M, Lp, field), mat_mul(Up, N, field)


# --------------------- Step 1b: symmetric reduction A = M P M^T (Lemma)
def reduce_to_symmetric_matching(A, field):
    """Return (M, singletons, spairs) with A = M P M^T, M invertible lower
    triangular, and P the symmetric matching matrix with
    P[t][t] = singletons[t] != 0, and for spairs[(i,j)] = (alpha, beta),
    i < j: P[i][j] = P[j][i] = alpha != 0, P[j][j] = beta."""
    n = len(A)
    M = [[field.one]]
    singletons = {}
    spairs = {}
    if A[0][0] != field.zero:
        singletons[0] = A[0][0]
    for k in range(1, n):
        a = [A[i][k] for i in range(k)]
        b = A[k][k]
        c = forward_solve(M, a, field)          # M^{-1} a
        # xi: blockwise solution of P xi = c on the nonzero blocks
        xi = [field.zero] * k
        for t, gamma in singletons.items():
            xi[t] = c[t] / gamma
        for (i, j), (alpha, beta) in spairs.items():
            xi[j] = c[i] / alpha
            xi[i] = (c[j] - beta * xi[j]) / alpha
        # P xi, residual e (supported on isolated zeros), corner f
        Pxi = [field.zero] * k
        for t, gamma in singletons.items():
            Pxi[t] = gamma * xi[t]
        for (i, j), (alpha, beta) in spairs.items():
            Pxi[i] = alpha * xi[j]
            Pxi[j] = alpha * xi[i] + beta * xi[j]
        e = [c[t] - Pxi[t] for t in range(k)]
        f = b
        for t in range(k):
            f = f - xi[t] * Pxi[t]              # f = b - xi^T P xi
        # grow M: last row is (xi, 1)
        for i in range(k):
            M[i].append(field.zero)
        M.append(xi + [field.one])
        istar = next((i for i in range(k) if e[i] != field.zero), None)
        if istar is not None:
            alpha = e[istar]
            for i in range(istar + 1, k):       # M <- M R: col istar update
                if e[i] != field.zero:
                    t = e[i] / alpha
                    for r in range(k + 1):
                        M[r][istar] = M[r][istar] + t * M[r][i]
            spairs[(istar, k)] = (alpha, f)
        elif f != field.zero:
            singletons[k] = f
    return M, singletons, spairs


# ------------- Step 3b: LDL^T of a symmetric matching matrix (threads)
def ldlt_of_symmetric_matching(singletons, spairs, n, field):
    """Given the symmetric matching matrix P, build (S, d) with
    S diag(d) S^T = P, S lower triangular, d a list of diagonal entries.
    Returns None iff the prefix condition chi_k <= zeta_k fails."""
    S = zeros(n, n, field)
    d = [field.zero] * n
    matched = set()
    for t, _ in singletons.items():
        matched.add(t)
    start, end = {}, {}
    for (i, j) in spairs:
        matched.add(i)
        matched.add(j)
        start[i], end[j] = (i, j), (i, j)
    threads = {}                                # root -> list of pairs
    free = []                                   # free thread roots (stack)
    busy = {}                                   # end index -> root
    for t in range(n):
        if t in end:
            free.append(busy.pop(t))            # thread becomes free at its end
        if t not in matched:
            threads[t] = []                     # isolated zero: new thread
            free.append(t)
        if t in start:
            if not free:
                return None                     # criterion fails
            root = free.pop()
            threads[root].append(start[t])
            busy[start[t][1]] = root
    # singletons: trivial one-column factors
    for t, gamma in singletons.items():
        S[t][t] = field.one
        d[t] = gamma
    # one telescoping chain of columns per thread
    for root, chain in threads.items():
        if not chain:
            continue                            # unused root: zero column
        m = len(chain)
        delta = field.one                       # delta_{m+1}
        s = [field.zero] * n                    # s_{m+1}
        cols = []                               # (index, column, d) per pair
        for q in range(m - 1, -1, -1):
            i, j = chain[q]
            alpha, beta = spairs[(i, j)]
            v = field.zero if beta != field.zero else field.one
            delta_q = delta * v * v - beta      # nonzero by construction
            lj = s[:]                           # ell_j = s_{q+1} + v e_j
            lj[j] = lj[j] + v
            s_q = [(delta * v / delta_q) * s[t] for t in range(n)]
            s_q[i] = s_q[i] - alpha / delta_q   # s_q
            li = s_q[:]                         # ell_i = s_q + e_j
            li[j] = li[j] + field.one
            cols.append((j, lj, delta))
            cols.append((i, li, -delta_q))
            delta, s = delta_q, s_q
        for t, col, dt in cols:
            for r in range(n):
                S[r][t] = col[r]
            d[t] = dt
        for r in range(n):                      # ell_root = s_1, d = delta_1
            S[r][root] = s[r]
        d[root] = delta
    return S, d


def ldlt_factorization(A, field):
    """A = L diag(d) L^T with L lower triangular (zeros allowed on the
    diagonal) and d diagonal, or None when no such factorization exists."""
    n = len(A)
    if n == 0:
        return [], []
    M, singletons, spairs = reduce_to_symmetric_matching(A, field)
    res = ldlt_of_symmetric_matching(singletons, spairs, n, field)
    if res is None:
        return None
    S, d = res
    return mat_mul(M, S, field), d
# scifig_end
