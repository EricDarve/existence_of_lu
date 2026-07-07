#!/usr/bin/env python3
"""
End-to-end numerical verification of the unsymmetric LU criterion (sufficiency)
for partial permutation matrices, plus the combinatorial equivalence.

For a partial permutation Pi satisfying condition (2):
    nullity(Pi[1:k,1:k]) <= nullity(Pi[:,1:k]) + nullity(Pi[1:k,:]^T)   for all k,
we build explicit L (lower-tri), U (upper-tri) via the largest-free-slot greedy
injection f: matched-pairs -> slots with f(i,j) <= min(i,j), and assert L@U == Pi.

We verify:
  - the three nullity formulas (F1,F2,F3) and the bookkeeping identity (3.3);
  - condition (2) <=> (chi^R_k <= c0_k AND chi^C_k <= r0_k);
  - for satisfying Pi: greedy injection exists, L lower-tri, U upper-tri, L@U == Pi
    (exact 0/1 integer arithmetic);
  - the forbidden swap [[0,1],[1,0]] FAILS (2);
  - several Pi with zero rows/cols that DO satisfy (2).
"""

import itertools
import numpy as np

rng = np.random.default_rng(20260607)


# ---------- exact integer rank over Q via fraction-free Gaussian elimination ----------
def rank_exact(M):
    """Exact rank of an integer matrix over Q using fraction-free (Bareiss-style) elim."""
    A = [[int(x) for x in row] for row in M]
    if not A:
        return 0
    m, n = len(A), len(A[0])
    rank = 0
    pr = 0
    for pc in range(n):
        # find pivot in column pc at or below row pr
        piv = None
        for r in range(pr, m):
            if A[r][pc] != 0:
                piv = r
                break
        if piv is None:
            continue
        A[pr], A[piv] = A[piv], A[pr]
        for r in range(m):
            if r != pr and A[r][pc] != 0:
                a = A[pr][pc]
                b = A[r][pc]
                # row_r = a*row_r - b*row_pr  (keeps integers; rank preserved)
                A[r] = [a * A[r][c] - b * A[pr][c] for c in range(n)]
        pr += 1
        rank += 1
        if pr == m:
            break
    return rank


def nullity_right(M):
    """Right nullity = ncols - rank."""
    M = np.asarray(M)
    ncols = M.shape[1]
    return ncols - rank_exact(M)


# ---------- partial permutation generation ----------
def random_partial_perm(n, density=0.7):
    """Random partial permutation: choose a random partial injection rows->cols."""
    Pi = np.zeros((n, n), dtype=int)
    cols = list(range(n))
    rng.shuffle(cols)
    used_cols = set()
    avail = list(range(n))
    rng.shuffle(avail)
    for i in range(n):
        if rng.random() < density:
            free = [c for c in range(n) if c not in used_cols]
            if not free:
                continue
            j = free[rng.integers(0, len(free))]
            Pi[i, j] = 1
            used_cols.add(j)
    return Pi


def all_partial_perms(n):
    """Enumerate all partial permutation matrices of size n."""
    cols_options = list(range(n)) + [None]  # None = zero row
    out = []
    for assignment in itertools.product(cols_options, repeat=n):
        used = [c for c in assignment if c is not None]
        if len(used) != len(set(used)):
            continue
        Pi = np.zeros((n, n), dtype=int)
        for i, c in enumerate(assignment):
            if c is not None:
                Pi[i, c] = 1
        out.append(Pi)
    return out


# ---------- combinatorial counts ----------
def matched_pairs(Pi):
    n = Pi.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            if Pi[i, j] == 1:
                pairs.append((i, j))  # 0-indexed
    return pairs


def counts(Pi, k):
    """k is a 1-based cut: indices 1..k correspond to 0-indexed 0..k-1."""
    pairs = matched_pairs(Pi)
    n = Pi.shape[0]
    # 0-indexed; cut at k means indices < k (0-based) are "<= k" (1-based)
    chiR = sum(1 for (i, j) in pairs if i < k <= j)   # i<=k<j  (1-based)  -> i<k? careful
    # Use 1-based throughout to avoid confusion:
    # convert pairs to 1-based
    p1 = [(i + 1, j + 1) for (i, j) in pairs]
    chiR = sum(1 for (i, j) in p1 if i <= k < j)
    chiC = sum(1 for (i, j) in p1 if j <= k < i)
    matched_rows = set(i for (i, j) in p1)
    matched_cols = set(j for (i, j) in p1)
    r0 = sum(1 for i in range(1, k + 1) if i not in matched_rows)
    c0 = sum(1 for j in range(1, k + 1) if j not in matched_cols)
    iota = sum(1 for (i, j) in p1 if i <= k and j <= k)
    return chiR, chiC, r0, c0, iota


def condition2(Pi):
    """Directly evaluate condition (2) via exact ranks."""
    n = Pi.shape[0]
    for k in range(1, n + 1):
        Akk = Pi[:k, :k]
        Ck = Pi[:, :k]          # n x k
        Rk = Pi[:k, :]          # k x n
        nA = nullity_right(Akk)            # k - rank
        nC = nullity_right(Ck)             # k - rank(C)
        nRt = nullity_right(Rk.T)          # k - rank(R)
        if nA > nC + nRt:
            return False
    return True


def combinatorial_ok(Pi):
    n = Pi.shape[0]
    for k in range(1, n + 1):
        chiR, chiC, r0, c0, iota = counts(Pi, k)
        if not (chiR <= c0 and chiC <= r0):
            return False
    return True


def formulas_ok(Pi):
    """Verify F1, F2, F3 and identity (3.3) hold for all k."""
    n = Pi.shape[0]
    for k in range(1, n + 1):
        chiR, chiC, r0, c0, iota = counts(Pi, k)
        nA = nullity_right(Pi[:k, :k])
        nC = nullity_right(Pi[:, :k])
        nRt = nullity_right(Pi[:k, :].T)
        if nA != r0 + chiR:
            return False
        if nA != c0 + chiC:
            return False
        if nC != c0:
            return False
        if nRt != r0:
            return False
        if r0 + chiR != c0 + chiC:
            return False
    return True


# ---------- explicit L,U via largest-free-slot greedy ----------
def build_LU(Pi):
    """
    Greedy injection f: pairs -> slots, f(i,j) <= min(i,j) (1-based),
    assigning the LARGEST free slot <= min(i,j). Then l_{f}=e_i, u_{f}=e_j.
    Returns (L, U) or None if it gets stuck (should never happen if (2) holds).
    """
    n = Pi.shape[0]
    pairs = [(i + 1, j + 1) for (i, j) in matched_pairs(Pi)]  # 1-based
    # order by nondecreasing min(i,j)
    pairs.sort(key=lambda p: min(p))
    free = set(range(1, n + 1))
    L = np.zeros((n, n), dtype=int)
    U = np.zeros((n, n), dtype=int)
    for (i, j) in pairs:
        m = min(i, j)
        candidates = [t for t in free if t <= m]
        if not candidates:
            return None  # stuck
        t = max(candidates)
        free.discard(t)
        # l_t = e_i  -> L[i-1, t-1] = 1 ; u_t = e_j -> U[t-1, j-1] = 1
        L[i - 1, t - 1] = 1
        U[t - 1, j - 1] = 1
    return L, U


def is_lower(L):
    n = L.shape[0]
    return all(L[i, j] == 0 for i in range(n) for j in range(n) if j > i)


def is_upper(U):
    n = U.shape[0]
    return all(U[i, j] == 0 for i in range(n) for j in range(n) if i > j)


# ====================== TESTS ======================
def main():
    results = {}

    # [A] nullity formulas on random partial perms
    fA = 0
    nA = 0
    for _ in range(4000):
        n = int(rng.integers(1, 9))
        Pi = random_partial_perm(n, density=float(rng.uniform(0.3, 1.0)))
        nA += 1
        if not formulas_ok(Pi):
            fA += 1
    results["[A] nullity formulas (random 4000)"] = (nA, fA)

    # [B] equivalence (2) <=> combinatorial, random
    fB = 0
    nB = 0
    for _ in range(4000):
        n = int(rng.integers(1, 9))
        Pi = random_partial_perm(n, density=float(rng.uniform(0.3, 1.0)))
        nB += 1
        if condition2(Pi) != combinatorial_ok(Pi):
            fB += 1
    results["[B] equivalence (2)<=>comb (random 4000)"] = (nB, fB)

    # [B2] exhaustive n=1..4
    fB2 = 0
    nB2 = 0
    fB2f = 0
    for n in range(1, 5):
        for Pi in all_partial_perms(n):
            nB2 += 1
            if condition2(Pi) != combinatorial_ok(Pi):
                fB2 += 1
            if not formulas_ok(Pi):
                fB2f += 1
    results["[B2] exhaustive n=1..4 equivalence"] = (nB2, fB2)
    results["[B2] exhaustive n=1..4 formulas"] = (nB2, fB2f)

    # [C] END-TO-END: for satisfying Pi build L,U and assert L@U==Pi, triangularity
    built = 0
    fC = 0
    stuck = 0
    nonsat_built = 0  # satisfying Pi where greedy returned None (should be 0)
    # random
    for _ in range(6000):
        n = int(rng.integers(1, 10))
        Pi = random_partial_perm(n, density=float(rng.uniform(0.2, 1.0)))
        if not condition2(Pi):
            continue
        built += 1
        LU = build_LU(Pi)
        if LU is None:
            stuck += 1
            fC += 1
            continue
        L, U = LU
        if not is_lower(L):
            fC += 1
            continue
        if not is_upper(U):
            fC += 1
            continue
        if not np.array_equal(L @ U, Pi):
            fC += 1
    results["[C] end-to-end L@U==Pi (random satisfying, %d built)" % built] = (built, fC)

    # [C2] exhaustive n=1..5: for every satisfying Pi build and check
    builtX = 0
    fCX = 0
    crossings = 0
    for n in range(1, 6):
        for Pi in all_partial_perms(n):
            if not condition2(Pi):
                # also confirm greedy is consistent: if not satisfying, we don't require build
                continue
            builtX += 1
            # detect a crossing index (above-root AND below-root)
            p1 = [(i + 1, j + 1) for (i, j) in matched_pairs(Pi)]
            above_roots = set(i for (i, j) in p1 if i < j)
            below_roots = set(j for (i, j) in p1 if i > j)
            if above_roots & below_roots:
                crossings += 1
            LU = build_LU(Pi)
            if LU is None:
                fCX += 1
                continue
            L, U = LU
            if not (is_lower(L) and is_upper(U) and np.array_equal(L @ U, Pi)):
                fCX += 1
    results["[C2] exhaustive n=1..5 end-to-end (%d satisfying, %d w/crossing)" % (builtX, crossings)] = (builtX, fCX)

    # [D] hand cases
    hand = {}

    swap = np.array([[0, 1], [1, 0]])
    hand["swap [[0,1],[1,0]] cond2(expect False)"] = condition2(swap)
    hand["swap combinatorial(expect False)"] = combinatorial_ok(swap)

    z3 = np.zeros((3, 3), dtype=int)
    hand["zero 3x3 cond2(expect True)"] = condition2(z3)

    I3 = np.eye(3, dtype=int)
    hand["I3 cond2(expect True)"] = condition2(I3)

    # crossing fixed by zero row+col: n=4, pairs (2,4) above and (4,2) below; rows/cols 1,3 zero
    cross = np.zeros((4, 4), dtype=int)
    cross[1, 3] = 1  # (2,4) 1-based
    cross[3, 1] = 1  # (4,2)
    hand["crossing-example cond2(expect True)"] = condition2(cross)
    hand["crossing-example combinatorial(expect True)"] = combinatorial_ok(cross)
    LU = build_LU(cross)
    if LU is not None:
        L, U = LU
        hand["crossing-example L@U==Pi"] = bool(np.array_equal(L @ U, cross) and is_lower(L) and is_upper(U))
        hand["crossing-example L"] = L.tolist()
        hand["crossing-example U"] = U.tolist()
    else:
        hand["crossing-example build"] = "STUCK"

    # full reversal n=3 (anti-diagonal) -> should FAIL
    rev3 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    hand["antidiag-3 cond2(expect False)"] = condition2(rev3)

    # above-arc rooted, zero col: pairs (1,3), zero col 1, zero row 2,3
    a1 = np.zeros((3, 3), dtype=int)
    a1[0, 2] = 1
    hand["above-arc(1,3) cond2(expect True)"] = condition2(a1)
    LU = build_LU(a1)
    if LU is not None:
        L, U = LU
        hand["above-arc(1,3) L@U==Pi"] = bool(np.array_equal(L @ U, a1) and is_lower(L) and is_upper(U))

    # below-arc (3,1), zero row 1, zero col 2,3
    b1 = np.zeros((3, 3), dtype=int)
    b1[2, 0] = 1
    hand["below-arc(3,1) cond2(expect True)"] = condition2(b1)
    LU = build_LU(b1)
    if LU is not None:
        L, U = LU
        hand["below-arc(3,1) L@U==Pi"] = bool(np.array_equal(L @ U, b1) and is_lower(L) and is_upper(U))

    # double crossing thread: n=6, pairs (1->5),(3->? ) make a U-thread
    # zero col 2, above arcs (1,5) and (3,6): chiR must be <= c0
    t1 = np.zeros((6, 6), dtype=int)
    t1[0, 4] = 1  # (1,5)
    t1[2, 5] = 1  # (3,6)
    # to satisfy, need 2 zero columns <= the cut; cols used: 5,6 -> cols 1,2,3,4 zero
    hand["U-thread two above-arcs cond2(expect True)"] = condition2(t1)
    LU = build_LU(t1)
    if LU is not None:
        L, U = LU
        hand["U-thread two above-arcs L@U==Pi"] = bool(np.array_equal(L @ U, t1) and is_lower(L) and is_upper(U))

    # ---------- report ----------
    print("=" * 70)
    print("UNSYMMETRIC LU CRITERION -- END-TO-END NUMERICAL VERIFICATION")
    print("=" * 70)
    total_fail = 0
    for name, (n, f) in results.items():
        total_fail += f
        print(f"  {name}: n={n} failures={f}")
    print("-" * 70)
    for k, v in hand.items():
        print(f"  {k}: {v}")
    print("-" * 70)
    print(f"TOTAL counted failures (random+exhaustive blocks): {total_fail}")

    # sanity asserts on hand cases
    assert hand["swap [[0,1],[1,0]] cond2(expect False)"] is False
    assert hand["swap combinatorial(expect False)"] is False
    assert hand["zero 3x3 cond2(expect True)"] is True
    assert hand["I3 cond2(expect True)"] is True
    assert hand["crossing-example cond2(expect True)"] is True
    assert hand["crossing-example L@U==Pi"] is True
    assert hand["antidiag-3 cond2(expect False)"] is False
    assert hand["above-arc(1,3) L@U==Pi"] is True
    assert hand["below-arc(3,1) L@U==Pi"] is True
    assert hand["U-thread two above-arcs cond2(expect True)"] is True
    assert hand["U-thread two above-arcs L@U==Pi"] is True
    assert total_fail == 0
    print("ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
