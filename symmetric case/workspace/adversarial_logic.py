"""
Scrutinize the LOGICAL claims of the proposed Step 2 + combinatorial part,
beyond the aggregate pass/fail of adversarial_verify.py.

Checks aimed at potential weak points:

(L1) The claim "rank of a 0/1 matrix with at most one 1 per row and per column
     equals its number of 1's" -- verify the leading block Pi_k really is such a
     matrix and rank == m_k EXACTLY (not just numerically).  Confirm rank == m_k
     over Q AND GF(2) on all exhaustive cases.

(L2) The Step-2 block identities rely on M[1:k,:] = (M_k | 0) and N[:,1:k]=(N_k;0).
     Verify these *structural* facts and that M_k, N_k invertible (nonzero diag).
     Edge: are they claiming M lower-tri => M_k same first k diagonal entries?
     Test that leading block of inverse-triangular product still works.

(L3) The proof reduces (2) to BOTH chiR_k<=c0_k AND chiC_k<=r0_k and claims each
     is *individually* equivalent to (2) at index k (via 3.3). Verify that
     for EVERY k individually:
         (nul_Ak <= nul_Acol + nul_ArowT)  <=>  (chiR_k <= c0_k)  <=> (chiC_k<=r0_k).
     This is a per-k claim, stronger than the global equivalence in Test B.

(L4) Potential trap: the proof's box says condition(2) for Pi  <=>  for all k both
     chiR_k<=c0_k and chiC_k<=r0_k.  But it argued each of (3.4),(3.5) is
     *equivalent* to (2) at k.  If (3.4)<=>(3.5) at each k (via 3.3), then "both"
     is redundant -- (2)_k <=> (3.4)_k alone.  Check (3.4)_k <=> (3.5)_k per k.
     If that fails for some k, the "AND" is genuinely needed and the per-k
     equivalence claim is WRONG.  Hunt for a counterexample.
"""
from fractions import Fraction
import random
random.seed(999)

def rank_Q(M):
    if not M: return 0
    A=[[Fraction(x) for x in r] for r in M]; nr=len(A); nc=len(A[0])
    pr=0; rk=0
    for c in range(nc):
        piv=None
        for r in range(pr,nr):
            if A[r][c]!=0: piv=r; break
        if piv is None: continue
        A[pr],A[piv]=A[piv],A[pr]; pv=A[pr][c]
        for r in range(nr):
            if r!=pr and A[r][c]!=0:
                f=A[r][c]/pv; A[r]=[A[r][j]-f*A[pr][j] for j in range(nc)]
        pr+=1; rk+=1
        if pr==nr: break
    return rk

def rank_GF2(M):
    if not M: return 0
    A=[[int(x)&1 for x in r] for r in M]; nr=len(A); nc=len(A[0])
    pr=0; rk=0
    for c in range(nc):
        piv=None
        for r in range(pr,nr):
            if A[r][c]: piv=r; break
        if piv is None: continue
        A[pr],A[piv]=A[piv],A[pr]
        for r in range(nr):
            if r!=pr and A[r][c]:
                A[r]=[A[r][j]^A[pr][j] for j in range(nc)]
        pr+=1; rk+=1
        if pr==nr: break
    return rk

def sub(M,rows,cols): return [[M[i][j] for j in cols] for i in rows]

def all_partial_perms(n):
    out=[]
    def rec(row,used,P):
        if row==n: out.append([r[:] for r in P]); return
        rec(row+1,used,P)
        for j in range(n):
            if j not in used:
                P[row][j]=1; rec(row+1,used|{j},P); P[row][j]=0
    rec(0,set(),[[0]*n for _ in range(n)]); return out

def combinatorial(P):
    n=len(P)
    matches=[(i,j) for i in range(n) for j in range(n) if P[i][j]==1]
    zr=[i for i in range(n) if all(P[i][j]==0 for j in range(n))]
    zc=[j for j in range(n) if all(P[i][j]==0 for i in range(n))]
    chiR=[0]*(n+1);chiC=[0]*(n+1);r0=[0]*(n+1);c0=[0]*(n+1);mk=[0]*(n+1)
    for k in range(1,n+1):
        chiR[k]=sum(1 for(i,j)in matches if (i+1)<=k<(j+1))
        chiC[k]=sum(1 for(i,j)in matches if (j+1)<=k<(i+1))
        r0[k]=sum(1 for i in zr if(i+1)<=k)
        c0[k]=sum(1 for j in zc if(j+1)<=k)
        mk[k]=sum(1 for(i,j)in matches if(i+1)<=k and(j+1)<=k)
    return chiR,chiC,r0,c0,mk

# --- L1: rank(Pi_k)==m_k exactly, over Q and GF2 ---
L1fail=0
for n in range(1,6):
    for P in all_partial_perms(n):
        chiR,chiC,r0,c0,mk=combinatorial(P)
        for k in range(1,n+1):
            Pk=sub(P,list(range(k)),list(range(k)))
            if rank_Q(Pk)!=mk[k] or rank_GF2(Pk)!=mk[k]:
                L1fail+=1
print(f"[L1] rank(Pi_k)==m_k over Q and GF2, exhaustive n<=5: failures={L1fail}")

# --- L3 & L4: per-k equivalences ---
# For each P and each k, compute the three booleans:
#   b2  = (nul_Pk <= nul_Pcol + nul_ProwT)   [condition (2) at k]
#   bR  = (chiR_k <= c0_k)                    [(3.4) at k]
#   bC  = (chiC_k <= r0_k)                    [(3.5) at k]
# Claim of proof: b2 <=> bR  and  b2 <=> bC   (each individually) at every k.
L3_b2_vs_bR_fail=0
L3_b2_vs_bC_fail=0
L4_bR_vs_bC_fail=0
examples_L4=[]
for n in range(1,6):
    for P in all_partial_perms(n):
        chiR,chiC,r0,c0,mk=combinatorial(P)
        for k in range(1,n+1):
            Pk=sub(P,list(range(k)),list(range(k)))
            Pcol=sub(P,list(range(n)),list(range(k)))
            Prow=sub(P,list(range(k)),list(range(n)))
            nul_Pk=k-rank_Q(Pk)
            nul_Pcol=k-rank_Q(Pcol)
            nul_ProwT=k-rank_Q(Prow)
            b2=(nul_Pk<=nul_Pcol+nul_ProwT)
            bR=(chiR[k]<=c0[k])
            bC=(chiC[k]<=r0[k])
            if b2!=bR: L3_b2_vs_bR_fail+=1
            if b2!=bC: L3_b2_vs_bC_fail+=1
            if bR!=bC:
                L4_bR_vs_bC_fail+=1
                if len(examples_L4)<5:
                    examples_L4.append((P,k,bR,bC,chiR[k],c0[k],chiC[k],r0[k]))
print(f"[L3] per-k:  (2)_k <=> (chiR_k<=c0_k):  failures={L3_b2_vs_bR_fail}")
print(f"[L3] per-k:  (2)_k <=> (chiC_k<=r0_k):  failures={L3_b2_vs_bC_fail}")
print(f"[L4] per-k:  (chiR_k<=c0_k) <=> (chiC_k<=r0_k):  mismatches={L4_bR_vs_bC_fail}")
for e in examples_L4:
    print("     [L4 counterexample-to-per-k-equivalence]", e)

# --- L2: structural facts about leading blocks of triangular M, N ---
# Verify M[1:k,:]=(M_k|0) and N[:,1:k]=(N_k;0) structurally on random triangular.
def rand_lower(n):
    return [[random.randint(-2,2) if j<i else (random.choice([1,-1,2]) if j==i else 0) for j in range(n)] for i in range(n)]
def rand_upper(n):
    return [[random.randint(-2,2) if j>i else (random.choice([1,-1,2]) if j==i else 0) for j in range(n)] for i in range(n)]
L2fail=0
for _ in range(2000):
    n=random.randint(1,7)
    M=rand_lower(n); N=rand_upper(n)
    for k in range(1,n+1):
        # M[1:k,:] columns beyond k must be zero
        for i in range(k):
            for j in range(k,n):
                if M[i][j]!=0: L2fail+=1
        # N[:,1:k] rows beyond k must be zero
        for i in range(k,n):
            for j in range(k):
                if N[i][j]!=0: L2fail+=1
print(f"[L2] structural M[1:k,:]=(M_k|0), N[:,1:k]=(N_k;0): failures={L2fail}")
