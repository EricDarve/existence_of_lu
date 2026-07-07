"""
DEEP structural adversarial test of proposed Step 1.

Beyond 'final product is correct', I verify EVERY structural CLAIM the proof prose makes,
exactly where it makes it, and I specifically hunt for cases that stress:

  (A) Stage 1 claim: after clearing matched-row entries, residual column e is supported
      ONLY on zero rows of Pi_{k-1}.  AND it does not disturb the leading block.
  (B) Stage 2 claim: symmetric for residual row g.
  (C) Stage 3 column-norm claim: 'row i* has its only nonzero entry, a 1, at column k'
      after scaling -- i.e. row i* is zero on cols 0..k-1.  Proof uses this to clear later
      zero-row entries WITHOUT disturbing the leading block.
  (D) Stage 3 row-norm claim: symmetric: 'col j* has only nonzero entry 1 at row k'.
  (E) The SIMULTANEOUS-match case: both a col-match (i*,k) and a row-match (k,j*) created.
      Then the proof's corner step uses the COLUMN branch (clear corner via row i*).
      Verify this does not damage the row-match at (k,j*).
      *** Potential danger: row_k <- row_k - w*row_i*.  Row i* has support {(i*,k)} only,
          so this only changes (k,k).  But does it touch (k, j*)?  row i* col j* must be 0.
          Since i* is a zero row of Pi_{k-1} and after column-norm row i* is supported only
          at col k, (i*, j*) = 0 for j* < k.  So (k, j*) is untouched.  VERIFY numerically. ***
  (F) Disturbance check: assert that EACH elementary op leaves the leading (k-1)x(k-1)
      block EXACTLY equal to Pi_{k-1} (the proof repeatedly claims 'changes nothing else
      in the leading block').

I deliberately generate matrices that FORCE simultaneous col+row matches and dense
matched structure.  Also: I verify that on a matrix that does NOT satisfy criterion (2)
the reduction STILL succeeds (Step 1 is unconditional -- it must work for ALL A; the
criterion only enters in Step 3).
"""
import random
from fractions import Fraction


class QField:
    name = "Q"
    def zero(self): return Fraction(0)
    def one(self): return Fraction(1)
    def add(self,a,b): return a+b
    def sub(self,a,b): return a-b
    def mul(self,a,b): return a*b
    def inv(self,a):
        if a==0: raise ZeroDivisionError
        return Fraction(1)/a
    def is_zero(self,a): return a==0
    def rand(self): return Fraction(random.randint(-2,2))
    def conv(self,x): return Fraction(x)

class GFp:
    def __init__(self,p): self.p=p; self.name=f"GF({p})"
    def zero(self): return 0
    def one(self): return 1%self.p
    def add(self,a,b): return (a+b)%self.p
    def sub(self,a,b): return (a-b)%self.p
    def mul(self,a,b): return (a*b)%self.p
    def inv(self,a):
        a%=self.p
        if a==0: raise ZeroDivisionError
        return pow(a,self.p-2,self.p)
    def is_zero(self,a): return a%self.p==0
    def rand(self): return random.randint(0,self.p-1)
    def conv(self,x): return x%self.p


def eye(F,n): return [[F.one() if i==j else F.zero() for j in range(n)] for i in range(n)]
def matmul(F,A,B):
    n=len(A); m=len(B[0]); kk=len(B)
    C=[[F.zero() for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for t in range(kk):
            a=A[i][t]
            if F.is_zero(a): continue
            for j in range(m):
                C[i][j]=F.add(C[i][j],F.mul(a,B[t][j]))
    return C
def mat_eq(F,A,B):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if not F.is_zero(F.sub(A[i][j],B[i][j])): return False
    return True


def reduce(F, A, stats):
    """Same construction, but assert EVERY structural claim. Increment stats counters
       when stressful cases occur (simultaneous match, etc.)."""
    n=len(A)
    W=[r[:] for r in A]; M=eye(F,n); N=eye(F,n); A0=[r[:] for r in A]

    def MWN_ok():
        return mat_eq(F, matmul(F,matmul(F,M,W),N), A0)

    def leading_block_snapshot(k):
        return [[W[i][j] for j in range(k)] for i in range(k)]

    def col_add(src,dst,lam):
        for i in range(n): W[i][dst]=F.add(W[i][dst],F.mul(lam,W[i][src]))
        for j in range(n): N[src][j]=F.sub(N[src][j],F.mul(lam,N[dst][j]))
    def row_add(src,dst,lam):
        for j in range(n): W[dst][j]=F.add(W[dst][j],F.mul(lam,W[src][j]))
        for i in range(n): M[i][src]=F.sub(M[i][src],F.mul(lam,M[i][dst]))
    def col_scale(idx,c):
        ci=F.inv(c)
        for i in range(n): W[i][idx]=F.mul(c,W[i][idx])
        for j in range(n): N[idx][j]=F.mul(N[idx][j],ci)
    def row_scale(idx,c):
        ci=F.inv(c)
        for j in range(n): W[idx][j]=F.mul(c,W[idx][j])
        for i in range(n): M[i][idx]=F.mul(M[i][idx],ci)

    for k in range(n):
        Pi_prev = leading_block_snapshot(k)   # = Pi_{k-1}
        row_match={}; col_match={}
        for i in range(k):
            js=[j for j in range(k) if not F.is_zero(W[i][j])]
            assert len(js)<=1
            if js: row_match[i]=js[0]; col_match[js[0]]=i
        zero_rows=[i for i in range(k) if i not in row_match]
        zero_cols=[j for j in range(k) if j not in col_match]

        # STAGE 1
        for (i,j) in list(row_match.items()):
            lam=W[i][k]
            if not F.is_zero(lam):
                col_add(j,k,F.sub(F.zero(),lam))
                # (F) leading block undisturbed
                assert mat_eq(F, leading_block_snapshot(k), Pi_prev), "stage1 disturbed leading block"
        # (A) residual col supported only on zero rows
        for i in range(k):
            if i in row_match:
                assert F.is_zero(W[i][k]), "(A) FAIL: matched-row residual nonzero"

        # STAGE 2
        for (j,i) in list(col_match.items()):
            lam=W[k][j]
            if not F.is_zero(lam):
                row_add(i,k,F.sub(F.zero(),lam))
                assert mat_eq(F, leading_block_snapshot(k), Pi_prev), "stage2 disturbed leading block"
        for j in range(k):
            if j in col_match:
                assert F.is_zero(W[k][j]), "(B) FAIL: matched-col residual nonzero"

        assert MWN_ok(), "invariant broken after stage2"

        # STAGE 3 column
        col_created=None
        e_nz=[i for i in zero_rows if not F.is_zero(W[i][k])]
        if e_nz:
            istar=e_nz[0]
            col_scale(k, F.inv(W[istar][k]))
            assert F.is_zero(F.sub(W[istar][k],F.one()))
            # (C) row istar zero on cols 0..k-1
            for j in range(k):
                assert F.is_zero(W[istar][j]), "(C) FAIL: row i* not zero on leading cols"
            for i in zero_rows:
                if i>istar and not F.is_zero(W[i][k]):
                    row_add(istar,i,F.sub(F.zero(),W[i][k]))
                    assert mat_eq(F, leading_block_snapshot(k), Pi_prev), "stage3col disturbed leading block"
            for i in range(k):
                if i==istar: assert F.is_zero(F.sub(W[i][k],F.one()))
                else: assert F.is_zero(W[i][k])
            col_created=istar

        # STAGE 3 row
        row_created=None
        g_nz=[j for j in zero_cols if not F.is_zero(W[k][j])]
        if g_nz:
            jstar=g_nz[0]
            row_scale(k, F.inv(W[k][jstar]))
            assert F.is_zero(F.sub(W[k][jstar],F.one()))
            # (D) col jstar zero on rows 0..k-1
            for i in range(k):
                assert F.is_zero(W[i][jstar]), "(D) FAIL: col j* not zero on leading rows"
            for j in zero_cols:
                if j>jstar and not F.is_zero(W[k][j]):
                    col_add(jstar,j,F.sub(F.zero(),W[k][j]))
                    assert mat_eq(F, leading_block_snapshot(k), Pi_prev), "stage3row disturbed leading block"
            for j in range(k):
                if j==jstar: assert F.is_zero(F.sub(W[k][j],F.one()))
                else: assert F.is_zero(W[k][j])
            row_created=jstar

        # (E) record simultaneous-match stress
        if col_created is not None and row_created is not None:
            stats['simul']+=1
            # snapshot the row-match position value BEFORE corner step
            jstar=row_created
            rowmatch_val_before = W[k][jstar]

        # CORNER
        w=W[k][k]
        if col_created is not None:
            istar=col_created
            if not F.is_zero(w):
                # verify row i* support is ONLY {(i*,k)} so this is safe
                supp=[c for c in range(n) if not F.is_zero(W[istar][c])]
                assert supp==[k], f"corner-safety FAIL: row i* support {supp} != [k]"
                row_add(istar,k,F.sub(F.zero(),w))
            assert F.is_zero(W[k][k]), "corner not cleared (col)"
            # (E) ensure row-match (if any) survived
            if row_created is not None:
                jstar=row_created
                assert F.is_zero(F.sub(W[k][jstar],F.one())), "(E) FAIL: corner step damaged row-match!"
        elif row_created is not None:
            jstar=row_created
            if not F.is_zero(w):
                supp=[r for r in range(n) if not F.is_zero(W[r][jstar])]
                assert supp==[k], f"corner-safety FAIL: col j* support {supp} != [k]"
                col_add(jstar,k,F.sub(F.zero(),w))
            assert F.is_zero(W[k][k]), "corner not cleared (row)"
        else:
            if not F.is_zero(w):
                col_scale(k, F.inv(w))
                assert F.is_zero(F.sub(W[k][k],F.one()))

        assert MWN_ok(), f"invariant broken after stage3 k={k}"
        # leading (k+1) partial perm
        for i in range(k+1):
            assert sum(1 for j in range(k+1) if not F.is_zero(W[i][j]))<=1
        for j in range(k+1):
            assert sum(1 for i in range(k+1) if not F.is_zero(W[i][j]))<=1

    assert MWN_ok()
    # M lower-tri invertible, N upper-tri invertible
    for i in range(n):
        for j in range(i+1,n): assert F.is_zero(M[i][j]), "M not lower"
        assert not F.is_zero(M[i][i]), "M singular"
        for j in range(i): assert F.is_zero(N[i][j]), "N not upper"
        assert not F.is_zero(N[i][i]), "N singular"
    # W partial perm 0/1
    for i in range(n):
        for j in range(n):
            v=W[i][j]
            if not F.is_zero(v):
                assert F.is_zero(F.sub(v,F.one())), "Pi entry not 0/1"
    return M,W,N


# ---- generators that FORCE stress ----
def rand_partial_perm(F,n):
    """random partial permutation, then SCRAMBLE by random lower/upper congruence to
       hide structure and force the reducer to recover it through all stages."""
    cols=list(range(n)); random.shuffle(cols)
    Pi=[[F.zero()]*n for _ in range(n)]
    for i in range(n):
        if random.random()<0.7:
            Pi[i][cols[i]]=F.one()
    # scramble: A = L Pi U with random invertible lower L, upper U
    L=eye(F,n); U=eye(F,n)
    for i in range(n):
        for j in range(i):
            if random.random()<0.5: L[i][j]=F.rand()
        for j in range(i+1,n):
            if random.random()<0.5: U[i][j]=F.rand()
    A=matmul(F,matmul(F,L,Pi),U)
    return A

def rand_full(F,n,dens):
    return [[ (F.rand() if random.random()<dens else F.zero()) for _ in range(n)] for _ in range(n)]


def main():
    random.seed(13371337)
    fields=[QField(),GFp(2),GFp(3),GFp(5),GFp(7),GFp(11)]
    grand_fail=0
    for F in fields:
        stats={'simul':0}
        passes=0
        for _ in range(800):
            n=random.randint(1,8)
            if random.random()<0.5:
                A=rand_partial_perm(F,n)
            else:
                A=rand_full(F,n,random.choice([0.3,0.6,1.0]))
            try:
                reduce(F,A,stats)
                passes+=1
            except AssertionError as ex:
                print(f"  [{F.name}] STRUCT FAIL n={n}: {ex}")
                for r in A: print("    ",[str(x) for x in r])
                grand_fail+=1
                break
        print(f"[{F.name:7s}] passes={passes:4d}  simultaneous-match-cases-seen={stats['simul']}")
    # explicit nasty hand cases incl. ones NOT satisfying criterion (2)
    print("--- hand cases (Step 1 must work UNCONDITIONALLY, incl. non-(2) matrices) ---")
    hand=[
        [[0,1],[1,0]],
        [[0,0,1],[1,0,0],[0,1,0]],            # full perm
        [[0,0,0],[0,0,1],[0,1,0]],
        [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]],  # two swaps
        [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]],
        [[1,1,0],[1,1,0],[0,0,0]],
        [[0,2,3],[4,0,6],[7,8,0]],
        [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
    ]
    for F in fields:
        ok=True
        for A in hand:
            Af=[[F.conv(x) for x in r] for r in A]
            try: reduce(F,Af,{'simul':0})
            except AssertionError as ex:
                print(f"  [{F.name}] HAND FAIL {A}: {ex}"); ok=False; grand_fail+=1
        print(f"[{F.name:7s}] hand {'OK' if ok else 'FAIL'}")
    print("GRAND FAIL:",grand_fail)

if __name__=="__main__":
    main()
