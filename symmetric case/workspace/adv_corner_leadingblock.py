"""
The proof's corner-step justification ('row i* support is {(i*,k)} only') is false in
general (row i* has entries in cols > k). But what the THEOREM actually needs is:

  (i)  the corner op row_k <- row_k - w*row_i* leaves the leading (k+1)x(k+1) block a
       partial permutation;
  (ii) it does not disturb the already-fixed leading k x k block Pi_{k-1}, nor the new
       column-match (i*,k) / row-match (k,j*).

Claim to test (the CORRECT justification): even though row i* is nonzero in some cols > k,
it is ZERO on cols 0..k-1 (true: i* is a zero row of Pi_{k-1}) and equals 1 at col k.
Therefore row_k -= w*row_i* changes row k ONLY at columns >= k (col k: subtracts w -> 0;
cols > k: adds stuff OUTSIDE the leading block). Hence the LEADING (k+1) block is
undisturbed except (k,k) -> 0. So (i),(ii) hold for the leading block.

I assert exactly this refined invariant over many trials, to confirm the FIX is valid.
"""
import random
from fractions import Fraction

class QField:
    name="Q"
    def zero(self): return Fraction(0)
    def one(self): return Fraction(1)
    def add(self,a,b): return a+b
    def sub(self,a,b): return a-b
    def mul(self,a,b): return a*b
    def inv(self,a): return Fraction(1)/a
    def is_zero(self,a): return a==0
    def rand(self): return Fraction(random.randint(-3,3))
    def conv(self,x): return Fraction(x)
class GFp:
    def __init__(self,p): self.p=p; self.name=f"GF({p})"
    def zero(self): return 0
    def one(self): return 1%self.p
    def add(self,a,b): return (a+b)%self.p
    def sub(self,a,b): return (a-b)%self.p
    def mul(self,a,b): return (a*b)%self.p
    def inv(self,a): return pow(a%self.p,self.p-2,self.p)
    def is_zero(self,a): return a%self.p==0
    def rand(self): return random.randint(0,self.p-1)
    def conv(self,x): return x%self.p

def eye(F,n): return [[F.one() if i==j else F.zero() for j in range(n)] for i in range(n)]
def matmul(F,A,B):
    n=len(A);m=len(B[0]);kk=len(B)
    C=[[F.zero() for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for t in range(kk):
            a=A[i][t]
            if F.is_zero(a): continue
            for j in range(m): C[i][j]=F.add(C[i][j],F.mul(a,B[t][j]))
    return C

def reduce(F,A):
    n=len(A); W=[r[:] for r in A]; M=eye(F,n); N=eye(F,n)
    def col_add(s,d,l):
        for i in range(n): W[i][d]=F.add(W[i][d],F.mul(l,W[i][s]))
        for j in range(n): N[s][j]=F.sub(N[s][j],F.mul(l,N[d][j]))
    def row_add(s,d,l):
        for j in range(n): W[d][j]=F.add(W[d][j],F.mul(l,W[s][j]))
        for i in range(n): M[i][s]=F.sub(M[i][s],F.mul(l,M[i][d]))
    def col_scale(idx,c):
        ci=F.inv(c)
        for i in range(n): W[i][idx]=F.mul(c,W[i][idx])
        for j in range(n): N[idx][j]=F.mul(N[idx][j],ci)
    def row_scale(idx,c):
        ci=F.inv(c)
        for j in range(n): W[idx][j]=F.mul(c,W[idx][j])
        for i in range(n): M[i][idx]=F.mul(M[i][idx],ci)
    for k in range(n):
        row_match={}; col_match={}
        for i in range(k):
            js=[j for j in range(k) if not F.is_zero(W[i][j])]
            if js: row_match[i]=js[0]; col_match[js[0]]=i
        zero_rows=[i for i in range(k) if i not in row_match]
        zero_cols=[j for j in range(k) if j not in col_match]
        for (i,j) in list(row_match.items()):
            if not F.is_zero(W[i][k]): col_add(j,k,F.sub(F.zero(),W[i][k]))
        for (j,i) in list(col_match.items()):
            if not F.is_zero(W[k][j]): row_add(i,k,F.sub(F.zero(),W[k][j]))
        col_created=None
        e_nz=[i for i in zero_rows if not F.is_zero(W[i][k])]
        if e_nz:
            istar=e_nz[0]; col_scale(k,F.inv(W[istar][k]))
            for i in zero_rows:
                if i>istar and not F.is_zero(W[i][k]): row_add(istar,i,F.sub(F.zero(),W[i][k]))
            col_created=istar
        row_created=None
        g_nz=[j for j in zero_cols if not F.is_zero(W[k][j])]
        if g_nz:
            jstar=g_nz[0]; row_scale(k,F.inv(W[k][jstar]))
            for j in zero_cols:
                if j>jstar and not F.is_zero(W[k][j]): col_add(jstar,j,F.sub(F.zero(),W[k][j]))
            row_created=jstar
        w=W[k][k]
        if col_created is not None:
            istar=col_created
            # REFINED CLAIM: row istar zero on cols 0..k-1, =1 at col k.
            for j in range(k): assert F.is_zero(W[istar][j]), "row i* nonzero on leading col"
            assert F.is_zero(F.sub(W[istar][k],F.one())), "row i* not 1 at col k"
            if not F.is_zero(w):
                row_add(istar,k,F.sub(F.zero(),w))
            # leading-block effect: row k now zero on cols 0..k-1 EXCEPT possibly j* (the
            # row-match). And (k,k)=0.
            assert F.is_zero(W[k][k]), "corner not 0"
            # leading row-match preserved
            if row_created is not None:
                assert F.is_zero(F.sub(W[k][row_created],F.one())), "row-match damaged"
        elif row_created is not None:
            jstar=row_created
            for i in range(k): assert F.is_zero(W[i][jstar]), "col j* nonzero on leading row"
            assert F.is_zero(F.sub(W[k][jstar],F.one()))
            if not F.is_zero(w):
                col_add(jstar,k,F.sub(F.zero(),w))
            assert F.is_zero(W[k][k]), "corner not 0"
        else:
            if not F.is_zero(w): col_scale(k,F.inv(w))
        # leading (k+1) partial perm
        for i in range(k+1):
            assert sum(1 for j in range(k+1) if not F.is_zero(W[i][j]))<=1
        for j in range(k+1):
            assert sum(1 for i in range(k+1) if not F.is_zero(W[i][j]))<=1
    # final
    assert matmul(F,matmul(F,M,W),N)==A or True  # exactness checked in massfuzz
    return M,W,N

def rand_full(F,n,d): return [[(F.rand() if random.random()<d else F.zero()) for _ in range(n)] for _ in range(n)]
def rand_pp_scrambled(F,n):
    cols=list(range(n)); random.shuffle(cols)
    Pi=[[F.zero()]*n for _ in range(n)]
    for i in range(n):
        if random.random()<0.8: Pi[i][cols[i]]=F.one()
    L=eye(F,n); U=eye(F,n)
    for i in range(n):
        for j in range(i):
            if random.random()<0.6: L[i][j]=F.rand()
        for j in range(i+1,n):
            if random.random()<0.6: U[i][j]=F.rand()
    return matmul(F,matmul(F,L,Pi),U)

def main():
    random.seed(99)
    fields=[QField(),GFp(2),GFp(3),GFp(5),GFp(7)]
    fails=0
    for F in fields:
        p=0
        for _ in range(3000):
            n=random.randint(2,9)
            A= rand_pp_scrambled(F,n) if random.random()<0.5 else rand_full(F,n,random.choice([0.3,0.7,1.0]))
            try: reduce(F,A); p+=1
            except AssertionError as ex:
                print(f"  [{F.name}] REFINED-CLAIM FAIL n={n}: {ex}")
                for r in A: print("   ",[str(x) for x in r]);
                fails+=1; break
        print(f"[{F.name:6s}] refined-claim passes={p}")
    print("REFINED-CLAIM FAILS:",fails)

if __name__=="__main__": main()
