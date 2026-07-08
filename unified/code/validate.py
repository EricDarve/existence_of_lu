"""Validation driver for matching_factorization.py.

Four independent layers of evidence, all in exact arithmetic:
  (1) randomized tests over Q: the constructions succeed exactly when the
      nullity criteria hold, and the factors multiply back exactly;
  (2) exhaustive tests over GF(2) and GF(3) for small n, cross-checked
      against BRUTE-FORCE enumeration of all triangular factorizations
      (this validates necessity independently of any of our proofs);
  (3) canonical-form check: the matching normal form computed by the
      reduction coincides with the rank-jump profile of A;
  (4) the symmetric corollary: for symmetric A the LU and LDL^T criteria
      agree, and the two constructions succeed or fail together.
"""

import itertools
import random
import sys
import time
from fractions import Fraction

from matching_factorization import (
    Fp, GFField, QField, forward_solve, is_lower, is_upper,
    ldlt_factorization, ldlt_of_symmetric_matching, lu_factorization,
    lu_of_matching, mat_mul, nullity_criterion_ldlt, nullity_criterion_lu,
    rank, reduce_to_matching, reduce_to_symmetric_matching, transpose, zeros)

Q = QField()
RESULTS = []


def report(name, count, failures):
    RESULTS.append((name, count, failures))
    print(f"  {name:<58s} cases = {count:>7d}   failures = {failures}")


# --------------------------------------------------------------- helpers
def check_lu_case(A, field):
    """Assert criterion <=> construction, and exactness of the factors."""
    n = len(A)
    crit = nullity_criterion_lu(A, field)
    res = lu_factorization(A, field)
    assert (res is not None) == crit, "criterion/construction mismatch (LU)"
    if res is not None:
        L, U = res
        assert is_lower(L, field) and is_upper(U, field), "not triangular"
        assert mat_mul(L, U, field) == A, "L U != A"
    return crit


def check_ldlt_case(A, field):
    n = len(A)
    crit = nullity_criterion_ldlt(A, field)
    res = ldlt_factorization(A, field)
    assert (res is not None) == crit, "criterion/construction mismatch (LDLT)"
    if res is not None:
        L, d = res
        assert is_lower(L, field), "L not lower triangular"
        LD = [[L[i][j] * d[j] for j in range(n)] for i in range(n)]
        assert mat_mul(LD, transpose(L), field) == A, "L D L^T != A"
    return crit


def matching_as_matrix(pairs, n, field):
    P = zeros(n, n, field)
    for (i, j), w in pairs.items():
        P[i][j] = w
    return P


def sym_matching_as_matrix(singletons, spairs, n, field):
    P = zeros(n, n, field)
    for t, g in singletons.items():
        P[t][t] = g
    for (i, j), (alpha, beta) in spairs.items():
        P[i][j] = alpha
        P[j][i] = alpha
        P[j][j] = beta
    return P


def check_reduction_invariants(A, field):
    """A = M Pi N with the stated structure, and Pi = rank-jump profile."""
    n = len(A)
    M, pairs, N = reduce_to_matching(A, field)
    P = matching_as_matrix(pairs, n, field)
    assert is_lower(M, field) and all(M[i][i] != field.zero for i in range(n))
    assert is_upper(N, field) and all(N[i][i] != field.zero for i in range(n))
    rows = [i for (i, j) in pairs]
    cols = [j for (i, j) in pairs]
    assert len(rows) == len(set(rows)) and len(cols) == len(set(cols))
    assert mat_mul(mat_mul(M, P, field), N, field) == A, "M Pi N != A"
    # canonical form: matched positions = jumps of the rank profile of A
    r = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            r[i][j] = rank([row[:j] for row in A[:i]], field)
    jumps = {(i - 1, j - 1) for i in range(1, n + 1) for j in range(1, n + 1)
             if r[i][j] - r[i - 1][j] - r[i][j - 1] + r[i - 1][j - 1] == 1}
    assert jumps == set(pairs), "matching != rank-jump profile"


def check_sym_reduction_invariants(A, field):
    n = len(A)
    M, singletons, spairs = reduce_to_symmetric_matching(A, field)
    P = sym_matching_as_matrix(singletons, spairs, n, field)
    assert is_lower(M, field) and all(M[i][i] != field.zero for i in range(n))
    assert mat_mul(mat_mul(M, P, field), transpose(M), field) == A
    r = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            r[i][j] = rank([row[:j] for row in A[:i]], field)
    jumps = {(i - 1, j - 1) for i in range(1, n + 1) for j in range(1, n + 1)
             if r[i][j] - r[i - 1][j] - r[i][j - 1] + r[i - 1][j - 1] == 1}
    expect = {(i, j) for (i, j) in spairs} | {(j, i) for (i, j) in spairs} \
        | {(t, t) for t in singletons}
    assert jumps == expect, "symmetric matching != rank-jump profile"


# ------------------------------------------------------- random matrices, Q
def rand_matrix(n, rng):
    style = rng.randrange(4)
    if style == 0:                               # dense, small integers
        return [[Fraction(rng.randint(-3, 3)) for _ in range(n)]
                for _ in range(n)]
    if style == 1:                               # sparse
        return [[Fraction(rng.randint(-2, 2)) if rng.random() > 0.65
                 else Fraction(0) for _ in range(n)] for _ in range(n)]
    if style == 2:                               # exact low rank
        r = rng.randrange(0, n + 1)
        B = [[Fraction(rng.randint(-2, 2)) for _ in range(r)]
             for _ in range(n)]
        C = [[Fraction(rng.randint(-2, 2)) for _ in range(n)]
             for _ in range(r)]
        return mat_mul(B, C, Q) if r else zeros(n, n, Q)
    T1 = [[Fraction(rng.randint(-2, 2)) if i > j else
           (Fraction(1) if i == j else Fraction(0)) for j in range(n)]
          for i in range(n)]
    T2 = [[Fraction(rng.randint(-2, 2)) if i < j else
           (Fraction(1) if i == j else Fraction(0)) for j in range(n)]
          for i in range(n)]
    S = [[Fraction(rng.randint(-1, 1)) if rng.random() > 0.5
          else Fraction(0) for _ in range(n)] for _ in range(n)]
    return mat_mul(T1, mat_mul(S, T2, Q), Q)     # hidden sparse core


def rand_sym_matrix(n, rng):
    style = rng.randrange(4)
    if style == 0:
        R = [[Fraction(rng.randint(-2, 2)) for _ in range(n)]
             for _ in range(n)]
        return [[R[i][j] + R[j][i] for j in range(n)] for i in range(n)]
    if style == 1:                               # sparse symmetric
        A = zeros(n, n, Q)
        for i in range(n):
            for j in range(i, n):
                v = Fraction(rng.randint(-2, 2)) if rng.random() > 0.6 \
                    else Fraction(0)
                A[i][j] = A[j][i] = v
        return A
    if style == 2:                               # exact low rank symmetric
        r = rng.randrange(0, n + 1)
        if r == 0:
            return zeros(n, n, Q)
        B = [[Fraction(rng.randint(-2, 2)) for _ in range(r)]
             for _ in range(n)]
        D = zeros(r, r, Q)
        for t in range(r):
            D[t][t] = Fraction(rng.choice([-2, -1, 1, 2]))
        return mat_mul(B, mat_mul(D, transpose(B), Q), Q)
    T = [[Fraction(rng.randint(-2, 2)) if i > j else
          (Fraction(1) if i == j else Fraction(0)) for j in range(n)]
         for i in range(n)]
    S = rand_sym_matrix_style1(n, rng)
    return mat_mul(T, mat_mul(S, transpose(T), Q), Q)


def rand_sym_matrix_style1(n, rng):
    A = zeros(n, n, Q)
    for i in range(n):
        for j in range(i, n):
            v = Fraction(rng.randint(-1, 1)) if rng.random() > 0.5 \
                else Fraction(0)
            A[i][j] = A[j][i] = v
    return A


# ------------------------------------------------- brute force over GF(p)
def all_lower(n, field, p):
    idx = [(i, j) for i in range(n) for j in range(i + 1)]
    for vals in itertools.product(range(p), repeat=len(idx)):
        L = zeros(n, n, field)
        for (i, j), v in zip(idx, vals):
            L[i][j] = field.of(v)
        yield L


def flat(A):
    return tuple(x.v for row in A for x in row)


def brute_force_lu_set(n, field, p):
    """All matrices L U with L lower and U upper triangular over GF(p)."""
    uppers = [transpose(L) for L in all_lower(n, field, p)]
    achievable = set()
    for L in all_lower(n, field, p):
        for U in uppers:
            achievable.add(flat(mat_mul(L, U, field)))
    return achievable


def brute_force_ldlt_set(n, field, p):
    """All matrices L D L^T over GF(p)."""
    achievable = set()
    for L in all_lower(n, field, p):
        Lt = transpose(L)
        for dv in itertools.product(range(p), repeat=n):
            LD = [[L[i][j] * field.of(dv[j]) for j in range(n)]
                  for i in range(n)]
            achievable.add(flat(mat_mul(LD, Lt, field)))
    return achievable


def all_matrices(n, field, p):
    for vals in itertools.product(range(p), repeat=n * n):
        yield [[field.of(vals[i * n + j]) for j in range(n)]
               for i in range(n)]


def all_sym_matrices(n, field, p):
    idx = [(i, j) for i in range(n) for j in range(i, n)]
    for vals in itertools.product(range(p), repeat=len(idx)):
        A = zeros(n, n, field)
        for (i, j), v in zip(idx, vals):
            A[i][j] = A[j][i] = field.of(v)
        yield A


# ----------------------------------------------------------------- tests
def test_special_cases():
    fails = 0
    swap = [[0, 1], [1, 0]]
    exch = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    anti3 = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    char2 = [[0, 1], [1, 1]]
    cases = [
        (swap, False), (exch, True), (anti3, False), (char2, False),
        ([[0] * 4 for _ in range(4)], True),
        ([[1 if i == j else 0 for j in range(4)] for i in range(4)], True),
    ]
    count = 0
    for A0, expected in cases:
        A = [[Fraction(v) for v in row] for row in A0]
        crit = check_lu_case(A, Q)
        assert crit == expected, f"unexpected LU verdict on {A0}"
        count += 1
        if A == transpose(A):
            crit2 = check_ldlt_case(A, Q)
            assert crit2 == expected
            count += 1
    F2 = GFField(2)
    for A0, expected in cases:
        A = [[F2.of(v) for v in row] for row in A0]
        assert check_lu_case(A, F2) == expected
        count += 1
    report("special cases (Q and GF(2))", count, fails)


def test_random_Q_lu(num=2000, seed=1):
    rng = random.Random(seed)
    sat = 0
    for _ in range(num):
        n = rng.randint(1, 7)
        A = rand_matrix(n, rng)
        if check_lu_case(A, Q):
            sat += 1
    report(f"random LU over Q ({sat} satisfying)", num, 0)


def test_random_Q_ldlt(num=2000, seed=2):
    rng = random.Random(seed)
    sat = 0
    for _ in range(num):
        n = rng.randint(1, 7)
        A = rand_sym_matrix(n, rng)
        if check_ldlt_case(A, Q):
            sat += 1
        # unified corollary: LU and LDL^T criteria agree for symmetric A
        assert nullity_criterion_lu(A, Q) == nullity_criterion_ldlt(A, Q)
    report(f"random LDLT over Q ({sat} satisfying)", num, 0)


def test_canonical_form(num=250, seed=3):
    rng = random.Random(seed)
    for _ in range(num):
        n = rng.randint(1, 6)
        check_reduction_invariants(rand_matrix(n, rng), Q)
        check_sym_reduction_invariants(rand_sym_matrix(n, rng), Q)
    report("canonical matching form = rank-jump profile (Q)", 2 * num, 0)


def test_exhaustive_gf2_lu():
    F2 = GFField(2)
    total = 0
    for n in range(1, 4):
        truth = brute_force_lu_set(n, F2, 2)
        for A in all_matrices(n, F2, 2):
            crit = check_lu_case(A, F2)
            assert crit == (flat(A) in truth), \
                "criterion disagrees with brute-force LU existence"
            total += 1
    report("exhaustive LU over GF(2), n<=3, vs brute force", total, 0)
    rng = random.Random(4)
    count = 5000
    for _ in range(count):
        n = rng.randint(4, 6)
        A = [[F2.of(rng.randrange(2)) for _ in range(n)] for _ in range(n)]
        check_lu_case(A, F2)
    report("random LU over GF(2), n=4..6", count, 0)


def test_exhaustive_gf2_ldlt():
    F2 = GFField(2)
    total = 0
    for n in range(1, 5):
        truth = brute_force_ldlt_set(n, F2, 2)
        for A in all_sym_matrices(n, F2, 2):
            crit = check_ldlt_case(A, F2)
            assert crit == (flat(A) in truth), \
                "criterion disagrees with brute-force LDLT existence"
            assert nullity_criterion_lu(A, F2) == crit
            total += 1
    report("exhaustive LDLT over GF(2), sym n<=4, vs brute force", total, 0)


def test_gf3():
    F3 = GFField(3)
    total = 0
    truth = brute_force_lu_set(2, F3, 3)
    for A in all_matrices(2, F3, 3):
        crit = check_lu_case(A, F3)
        assert crit == (flat(A) in truth)
        total += 1
    for A in all_matrices(3, F3, 3):
        check_lu_case(A, F3)
        total += 1
    report("exhaustive LU over GF(3), n=2 brute force + n=3", total, 0)
    total = 0
    truth = brute_force_ldlt_set(3, F3, 3)
    for A in all_sym_matrices(3, F3, 3):
        crit = check_ldlt_case(A, F3)
        assert crit == (flat(A) in truth)
        assert nullity_criterion_lu(A, F3) == crit
        total += 1
    report("exhaustive LDLT over GF(3), sym n<=3, vs brute force", total, 0)


def main():
    t0 = time.time()
    print("validating unpivoted LU / LDLT via matching normal forms")
    test_special_cases()
    test_random_Q_lu()
    test_random_Q_ldlt()
    test_canonical_form()
    test_exhaustive_gf2_lu()
    test_exhaustive_gf2_ldlt()
    test_gf3()
    bad = sum(f for _, _, f in RESULTS)
    print(f"TOTAL: {sum(c for _, c, _ in RESULTS)} cases, "
          f"{bad} failures, {time.time() - t0:.1f}s")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
