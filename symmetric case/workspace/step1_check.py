"""
Numerical verification of STEP 1 (unsymmetric LPU / generalized Bruhat):

    Every A in F^{n x n} factors as  A = M Pi N,
    with M invertible lower-triangular, N invertible upper-triangular,
    and Pi a partial permutation matrix (0/1, at most one 1 per row and per col).

Implementation strategy.
We maintain explicit M (inv lower-tri) and N (inv upper-tri), built up so that
at every stage the leading k x k block of  Pi := M^{-1} A N^{-1}  is a partial
permutation matrix.  At the end Pi is a partial permutation and A = M Pi N.

The inductive step k-1 -> k operates on the *reduced* bordered matrix
        Pi_{k-1} = M_{k-1}^{-1} A_{k-1} N_{k-1}^{-1}.
The new reduced column is  cc = M_{k-1}^{-1} a   (forward solve, M lower-tri),
the new reduced row    is  rr = (N_{k-1}^{-1})^T-ish row  c^T N_{k-1}^{-1}
                              = solve  N_{k-1}^T x = c  (N^T lower-tri) -> rr = x.

Column operations that ADD a multiple of an earlier column (index < k) to the
LAST column k are upper-triangular and are recorded in N (they multiply A on the
right by an upper-tri elementary, so N_new = N_old * E with E upper-tri, but we
record the *accumulated* N directly).  Row operations that ADD a multiple of an
earlier row (index < k) to the LAST row k are lower-triangular and recorded in M.

We work directly with the bordered reduced matrix and keep M, N as the change of
basis.  Concretely we store M, N as full matrices and, at each step, compute the
reduced column/row, perform eliminations expressed as updates to M and N, and
let Pi be re-derived at the very end as M^{-1} A N^{-1}.

To make the bookkeeping transparent and *provably* correct we instead carry a
"reduced" working matrix  W  (which always equals  M^{-1} A N^{-1}  on the
leading block) and apply the elementary operations to W, M, N simultaneously.
"""

import random
from fractions import Fraction

# ----------------------------------------------------------------------
# Field abstraction
# ----------------------------------------------------------------------

class Field:
    def __init__(self, name, zero, one, add, sub, mul, inv, eq, rand):
        self.name = name
        self.zero = zero; self.one = one
        self.add = add; self.sub = sub; self.mul = mul; self.inv = inv
        self.eq = eq; self.rand = rand

def field_Q():
    F0, F1 = Fraction(0), Fraction(1)
    return Field("Q", F0, F1,
        lambda a,b: a+b, lambda a,b: a-b, lambda a,b: a*b,
        lambda a: F1/a, lambda a,b: a==b,
        lambda: Fraction(random.randint(-3,3), random.randint(1,3)))

def field_GF(p):
    def inv(a): return pow(a % p, p-2, p)
    return Field(f"GF({p})", 0, 1,
        lambda a,b:(a+b)%p, lambda a,b:(a-b)%p, lambda a,b:(a*b)%p,
        inv, lambda a,b:(a-b)%p==0, lambda: random.randrange(p))

# ----------------------------------------------------------------------
# Matrix helpers
# ----------------------------------------------------------------------

def zeros(F,m,n): return [[F.zero]*n for _ in range(m)]
def identity(F,n):
    I=zeros(F,n,n)
    for i in range(n): I[i][i]=F.one
    return I
def matmul(F,A,B):
    m=len(A); kk=len(B); n=len(B[0]) if B else 0
    C=zeros(F,m,n)
    for i in range(m):
        Ai=A[i]; Ci=C[i]
        for t in range(kk):
            a=Ai[t]
            if F.eq(a,F.zero): continue
            Bt=B[t]
            for j in range(n):
                Ci[j]=F.add(Ci[j],F.mul(a,Bt[j]))
    return C
def mat_eq(F,A,B):
    if len(A)!=len(B): return False
    for ra,rb in zip(A,B):
        for a,b in zip(ra,rb):
            if not F.eq(a,b): return False
    return True
def is_lower_tri(F,A):
    n=len(A)
    return all(F.eq(A[i][j],F.zero) for i in range(n) for j in range(i+1,n))
def is_upper_tri(F,A):
    n=len(A)
    return all(F.eq(A[i][j],F.zero) for i in range(n) for j in range(i))
def diag_nonzero(F,A):
    return all(not F.eq(A[i][i],F.zero) for i in range(len(A)))
def is_partial_perm(F,P):
    n=len(P)
    for i in range(n):
        for j in range(n):
            v=P[i][j]
            if not (F.eq(v,F.zero) or F.eq(v,F.one)): return False
    for i in range(n):
        if sum(1 for j in range(n) if F.eq(P[i][j],F.one))>1: return False
    for j in range(n):
        if sum(1 for i in range(n) if F.eq(P[i][j],F.one))>1: return False
    return True
def solve_lower(F,L,b):
    n=len(L); x=[F.zero]*n
    for i in range(n):
        s=b[i]
        for j in range(i): s=F.sub(s,F.mul(L[i][j],x[j]))
        x[i]=F.mul(s,F.inv(L[i][i]))
    return x

# ----------------------------------------------------------------------
# STEP 1.
#
# We keep:
#   W : the reduced working matrix; the leading k x k block of W is a partial
#       permutation matrix and equals  M^{-1} A N^{-1}  there.  We store W full
#       (n x n) but only the leading block is meaningful / maintained.
#   M : invertible lower-tri, N : invertible upper-tri, with
#       A_k = M_k W_k N_k  for the leading blocks.
#
# Elementary operations (all act on the leading bordered block of size k):
#  (col op, upper-tri) add lambda*col(j) to col(k), j<k:
#       this is  W <- W E,  A unchanged,  N <- E^{-1} N? -- careful with sides.
#
# We instead maintain the invariant  A = M W N  globally by applying *inverse*
# operations to M or N.  Let us define operations on (W, M, N):
#
#  RowAddToLast(i, lam):  add lam * (row i) to row k of W's leading block.
#     In matrix form W <- E W with E = I + lam e_k e_i^T (lower-tri since i<k).
#     To preserve A = M W N we set M <- M E^{-1} = M (I - lam e_k e_i^T).
#     E^{-1} lower-tri, so M stays lower-tri.  (M e_k e_i^T affects only col i of
#     M using col k of M; since k is the new last index, M[:,k]=e_k currently,
#     so M <- M - lam M[:,k] e_i^T  subtracts lam e_k from column i... that would
#     put a nonzero at (k,i) which is fine, lower-tri.)  GOOD.
#
#  ColAddToLast(j, lam):  add lam * (col j) to col k of W's leading block.
#     W <- W E,  E = I + lam e_j e_k^T (upper-tri since j<k).
#     A = M W N preserved by N <- E^{-1} N = (I - lam e_j e_k^T) N.
#     Affects row j of N using row k of N; N[k,:]=e_k currently, fine, upper-tri.
#
#  ScaleLastRow(s): multiply row k of W by s (s != 0). W <- D W, D=I+(s-1)e_k e_k^T
#     M <- M D^{-1}: scales column k of M by 1/s. lower-tri preserved.
#
#  ScaleLastCol(s): multiply col k of W by s. W <- W D. N <- D^{-1} N scales
#     row k of N by 1/s. upper-tri preserved.
#
# These four ops are enough.  We implement them as direct updates.
# ----------------------------------------------------------------------

class State:
    def __init__(self, F, A):
        self.F=F; self.n=len(A)
        self.A=[row[:] for row in A]
        self.W=[row[:] for row in A]          # will be reduced in place
        self.M=identity(F,self.n)
        self.N=identity(F,self.n)

    def row_add_to_last(self, k, i, lam):
        # add lam*row(i) to row(k) of W (leading block indices), i<k
        F=self.F
        if F.eq(lam,F.zero): return
        for c in range(self.n):
            self.W[k][c]=F.add(self.W[k][c],F.mul(lam,self.W[i][c]))
        # M <- M (I - lam e_k e_i^T):  col i of M  -=  lam * col k of M
        for r in range(self.n):
            self.M[r][i]=F.sub(self.M[r][i],F.mul(lam,self.M[r][k]))

    def col_add_to_last(self, k, j, lam):
        # add lam*col(j) to col(k) of W, j<k
        F=self.F
        if F.eq(lam,F.zero): return
        for r in range(self.n):
            self.W[r][k]=F.add(self.W[r][k],F.mul(lam,self.W[r][j]))
        # N <- (I - lam e_j e_k^T) N:  row j of N  -=  lam * row k of N
        for c in range(self.n):
            self.N[j][c]=F.sub(self.N[j][c],F.mul(lam,self.N[k][c]))

    def scale_last_row(self, k, s):
        F=self.F
        if F.eq(s,F.one): return
        for c in range(self.n):
            self.W[k][c]=F.mul(s,self.W[k][c])
        # M col k *= 1/s
        si=F.inv(s)
        for r in range(self.n):
            self.M[r][k]=F.mul(self.M[r][k],si)

    def scale_last_col(self, k, s):
        F=self.F
        if F.eq(s,F.one): return
        for r in range(self.n):
            self.W[r][k]=F.mul(s,self.W[r][k])
        si=F.inv(s)
        for c in range(self.n):
            self.N[k][c]=F.mul(self.N[k][c],si)


def step1(F, A):
    n=len(A)
    st=State(F,A)
    W=st.W
    # matched structure of leading block as we go
    matched_row=[None]*n   # row i -> col j
    matched_col=[None]*n   # col j -> row i

    for k in range(n):     # process index k (zero-based) -> leading block size k+1
        # At entry, leading k x k (indices 0..k-1) block of W is a partial perm,
        # consistent with matched_row/matched_col.  The new row k and col k of W
        # currently hold A's reduced data ONLY IF we first reduce them.  But W
        # was initialized to A and earlier ops already touched rows/cols >= via
        # row_add/col_add? No: our ops only modify the LAST processed row/col k
        # and earlier rows/cols of M,N; W earlier rows/cols (the leading block)
        # are partial-perm.  However W[k][*] and W[*][k] still hold raw A mixed
        # with the basis changes applied so far.  Because every op we applied so
        # far modified only rows<=current and cols<=current of W, the entries
        # W[k][0..k-1], W[0..k-1][k], W[k][k] are exactly the *reduced* border:
        #   W[k][j] = (M^{-1} A N^{-1})[k][j].
        # (This holds because M,N are triangular with unit-updated structure and
        #  the leading principal reduction is independent of later indices.)
        #
        # ------- Step (1): clear new-column entries in MATCHED rows -------
        # For i<k matched to col j (matched_row[i]=j), entry W[i][k] is killed by
        # col op: col(k) -= W[i][k] * col(j)   (because W[i][j]=1).  Upper-tri.
        for i in range(k):
            j=matched_row[i]
            if j is not None and not F.eq(W[i][k],F.zero):
                lam=F.sub(F.zero,W[i][k])  # add (-W[i][k])*col(j)
                st.col_add_to_last(k,j,lam)
        # Now residual column W[0..k-1][k] supported only on UNMATCHED (zero)
        # rows i (matched_row[i] is None).

        # ------- Step (2): clear new-row entries in MATCHED columns -------
        for j in range(k):
            i=matched_col[j]
            if i is not None and not F.eq(W[k][j],F.zero):
                lam=F.sub(F.zero,W[k][j])  # add (-W[k][j])*row(i)
                st.row_add_to_last(k,i,lam)
        # residual row W[k][0..k-1] supported only on unmatched (zero) cols.

        # ------- Step (3): normalize residual column -> new matched (i*,k) -----
        # residual column entries live on zero-rows.  Find first nonzero.
        i_star=None
        for i in range(k):
            if matched_row[i] is None and not F.eq(W[i][k],F.zero):
                i_star=i; break
        if i_star is not None:
            alpha=W[i_star][k]
            # scale col k so W[i_star][k] = 1
            st.scale_last_col(k,F.inv(alpha))
            # now clear the remaining entries below i_star in column k that sit
            # on zero-rows: use ROW ops adding row i_star to those rows? No --
            # we must clear W[i][k] for i>i_star (i<k) on zero rows.  Subtracting
            # a multiple of row i_star (which has its 1 now at (i_star,k)) from
            # row i:  row(i) -= W[i][k]*row(i_star).  But i_star<i and i<k, and
            # this modifies an EARLIER row i (< k), i.e. a row inside the leading
            # block.  That is a lower-tri row op on rows of the leading block; it
            # is recorded in M as a lower-tri update (i_star<i).  Implement via a
            # general row-add (not only to last).  We add helper below.
            for i in range(i_star+1,k):
                if matched_row[i] is None and not F.eq(W[i][k],F.zero):
                    lam=F.sub(F.zero,W[i][k])
                    _row_add_general(st,i,i_star,lam)   # row(i)+=lam*row(i_star)
            # record the new match (i_star, k)
            matched_row[i_star]=k
            matched_col[k]=i_star

        # ------- Step (3'): normalize residual row -> new matched (k,j*) -----
        j_star=None
        for j in range(k):
            if matched_col[j] is None and not F.eq(W[k][j],F.zero):
                j_star=j; break
        if j_star is not None:
            beta=W[k][j_star]
            st.scale_last_row(k,F.inv(beta))
            for j in range(j_star+1,k):
                if matched_col[j] is None and not F.eq(W[k][j],F.zero):
                    lam=F.sub(F.zero,W[k][j])
                    _col_add_general(st,j,j_star,lam)   # col(j)+=lam*col(j_star)
            matched_col[j_star]=k
            matched_row[k]=j_star

        # ------- corner reconciliation -------
        # After the above, W[k][k] may be nonzero.  Cases:
        #  (a) both i_star and j_star found: row k and col k each carry a 1 at an
        #      earlier index. The corner W[k][k] must be cleared. Clear it using
        #      the col-match: subtract W[k][k]*col(... ) -- the 1 in row k is at
        #      (k, j_star); the 1 in col k is at (i_star, k). To kill W[k][k]
        #      without disturbing the placed 1s, add  -W[k][k]*row(i_star) to
        #      row(k): row(k) has its 1 at (k,j_star); row(i_star) has support
        #      only at (i_star,k). So row(k) -= W[k][k]*row(i_star) changes
        #      W[k][k] -= W[k][k]*W[i_star][k] = W[k][k]*1 -> 0, and adds
        #      -W[k][k] at (k,?) where row(i_star) nonzero = only col k. Also
        #      W[k][j_star] unaffected (row i_star zero there). i_star<k: lower
        #      tri row op (record in M). GOOD.
        #  (b) only i_star: row k has no placed 1 (j_star None). corner sits at
        #      (k,k) of col k which already has a 1 at (i_star,k). Clear corner by
        #      row(k) -= W[k][k]*row(i_star) as above. Leaves row k zero => index
        #      k is a zero row. fine.
        #  (c) only j_star: symmetric, clear corner via col(k) -= W[k][k]*col(j_star).
        #  (d) neither: corner alone. If W[k][k]!=0 scale to 1 -> matched (k,k);
        #      else isolated zero.
        if i_star is not None or j_star is not None:
            if not F.eq(W[k][k],F.zero):
                if i_star is not None:
                    lam=F.sub(F.zero,W[k][k])
                    st.row_add_to_last(k,i_star,lam)
                else:
                    lam=F.sub(F.zero,W[k][k])
                    st.col_add_to_last(k,j_star,lam)
        else:
            if not F.eq(W[k][k],F.zero):
                st.scale_last_col(k,F.inv(W[k][k]))
                matched_row[k]=k; matched_col[k]=k
        # leading (k+1) block of W now a partial perm.

    # Pi = leading n x n of W (all of it)
    Pi=[row[:] for row in st.W]
    return st.M, Pi, st.N


# general row/col adds that touch earlier rows/cols of the leading block.
def _row_add_general(st, dst, src, lam):
    # row(dst) += lam * row(src), with src<dst (lower-tri).  W,M update.
    F=st.F
    if F.eq(lam,F.zero): return
    for c in range(st.n):
        st.W[dst][c]=F.add(st.W[dst][c],F.mul(lam,st.W[src][c]))
    # A=MWN preserved: W<-E W, E=I+lam e_dst e_src^T (lower-tri, src<dst).
    # M <- M E^{-1} = M (I - lam e_dst e_src^T): col src of M -= lam*col dst of M.
    for r in range(st.n):
        st.M[r][src]=F.sub(st.M[r][src],F.mul(lam,st.M[r][dst]))

def _col_add_general(st, dst, src, lam):
    # col(dst) += lam*col(src), src<dst (upper-tri).
    F=st.F
    if F.eq(lam,F.zero): return
    for r in range(st.n):
        st.W[r][dst]=F.add(st.W[r][dst],F.mul(lam,st.W[r][src]))
    # W<-W E, E=I+lam e_src e_dst^T (upper-tri). N <- E^{-1} N:
    # row src of N -= lam*row dst of N.
    for c in range(st.n):
        st.N[src][c]=F.sub(st.N[src][c],F.mul(lam,st.N[dst][c]))


# ----------------------------------------------------------------------
def random_matrix(F,n,density=0.7):
    A=zeros(F,n,n)
    for i in range(n):
        for j in range(n):
            if random.random()<density:
                A[i][j]=F.rand()
    return A

def check_one(F,A):
    M,Pi,N=step1(F,A)
    prod=matmul(F,matmul(F,M,Pi),N)
    if not mat_eq(F,prod,A): return False,"A != M Pi N"
    if not (is_lower_tri(F,M) and diag_nonzero(F,M)): return False,"M not inv lower-tri"
    if not (is_upper_tri(F,N) and diag_nonzero(F,N)): return False,"N not inv upper-tri"
    if not is_partial_perm(F,Pi): return False,"Pi not partial perm"
    return True,""

def run_tests(F,ntests=400,nmax=7):
    passes=fails=0
    for _ in range(ntests):
        n=random.randint(1,nmax)
        A=random_matrix(F,n,density=random.choice([0.2,0.4,0.6,0.8,1.0]))
        ok,msg=check_one(F,A)
        if ok: passes+=1
        else:
            fails+=1
            if fails<=4:
                print(f"  FAIL ({F.name}) n={n}: {msg}")
                print("   A=",A)
                M,Pi,N=step1(F,A)
                print("   M=",M);print("   Pi=",Pi);print("   N=",N)
    print(f"[{F.name}] passes={passes} fails={fails}")
    return fails

if __name__=="__main__":
    random.seed(1)
    tf=0
    tf+=run_tests(field_Q(),400,7)
    tf+=run_tests(field_GF(2),500,8)
    tf+=run_tests(field_GF(3),500,8)
    tf+=run_tests(field_GF(5),500,8)
    tf+=run_tests(field_GF(7),500,8)
    F=field_Q()
    hand=[
        [[Fraction(0)]],
        [[Fraction(0),Fraction(1)],[Fraction(1),Fraction(0)]],
        [[Fraction(0),Fraction(0)],[Fraction(0),Fraction(0)]],
        [[Fraction(0),Fraction(0),Fraction(0)],
         [Fraction(0),Fraction(0),Fraction(1)],
         [Fraction(0),Fraction(1),Fraction(0)]],
        [[Fraction(1),Fraction(2),Fraction(3)],
         [Fraction(4),Fraction(5),Fraction(6)],
         [Fraction(7),Fraction(8),Fraction(0)]],
    ]
    print("hand cases:")
    for A in hand:
        ok,msg=check_one(F,A)
        print(f"   {'OK ' if ok else 'FAIL '+msg}  A={[[str(x) for x in r] for r in A]}")
        if not ok: tf+=1
    print("TOTAL FAILS:",tf)
