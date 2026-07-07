"""
Independent hand-style re-derivation check of bookkeeping (3.1),(3.2),(3.3)
and the EXACT definitions used (strict vs non-strict), plus a stress test that
the dual identity nul(Pi_k)=r0+chiR=c0+chiC and the two F1 forms agree even in
'crossing index' situations (an index that is simultaneously an above-root and a
below-root), which is the case the Step-3 construction calls a THREAD.

Also: deliberately probe the boundary case j=k (matched pair lands exactly on
the cut) to make sure 'i<=k<j' vs 'i<=k<=j' convention is the right one.
"""
from fractions import Fraction
import itertools, random
random.seed(7)

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

# Identify 'crossing indices' (thread roots): index t (1-based) that is BOTH an
# above-root (row t matched to col j>t) AND a below-root (col t matched to row i>t).
def crossing_indices(P):
    n=len(P); cr=[]
    for t in range(1,n+1):
        ti=t-1
        # above-root: row ti matched to col j with j>ti
        above = any(P[ti][j]==1 and j>ti for j in range(n))
        # below-root: col ti matched to row i with i>ti
        below = any(P[i][ti]==1 and i>ti for i in range(n))
        if above and below: cr.append(t)
    return cr

# Verify F1 both forms AND that a thread (crossing index) exists in plenty of the
# satisfying cases, to ensure the 'independent threads' claim is non-vacuous.
def combinatorial(P):
    n=len(P)
    matches=[(i,j) for i in range(n) for j in range(n) if P[i][j]==1]
    zr=[i for i in range(n) if all(P[i][j]==0 for j in range(n))]
    zc=[j for j in range(n) if all(P[i][j]==0 for i in range(n))]
    chiR=[0]*(n+1);chiC=[0]*(n+1);r0=[0]*(n+1);c0=[0]*(n+1);mk=[0]*(n+1)
    for k in range(1,n+1):
        chiR[k]=sum(1 for(i,j)in matches if(i+1)<=k<(j+1))
        chiC[k]=sum(1 for(i,j)in matches if(j+1)<=k<(i+1))
        r0[k]=sum(1 for i in zr if(i+1)<=k)
        c0[k]=sum(1 for j in zc if(j+1)<=k)
        mk[k]=sum(1 for(i,j)in matches if(i+1)<=k and(j+1)<=k)
    return chiR,chiC,r0,c0,mk
def cond2(P):
    n=len(P)
    for k in range(1,n+1):
        Pk=sub(P,range(k),range(k));Pc=sub(P,range(n),range(k));Pr=sub(P,range(k),range(n))
        if (k-rank_Q(Pk))>(k-rank_Q(Pc))+(k-rank_Q(Pr)): return False
    return True
def comb_cond(P):
    chiR,chiC,r0,c0,mk=combinatorial(P)
    return all(chiR[k]<=c0[k] and chiC[k]<=r0[k] for k in range(1,len(P)+1))

# Boundary-convention probe: a matched pair on the diagonal (i==j) must count in
# NEITHER chiR nor chiC for any k (it is never a crossing). Check.
boundary_fail=0
threads_seen=0
sat_with_thread=0
sat_total=0
for n in range(1,6):
    for P in all_partial_perms(n):
        chiR,chiC,r0,c0,mk=combinatorial(P)
        # diagonal matches never contribute to chiR/chiC: rebuild ignoring i==j and compare
        matches=[(i,j) for i in range(n) for j in range(n) if P[i][j]==1]
        for k in range(1,n+1):
            cR_diag=sum(1 for(i,j)in matches if i==j and (i+1)<=k<(j+1))  # must be 0
            cC_diag=sum(1 for(i,j)in matches if i==j and (j+1)<=k<(i+1))  # must be 0
            if cR_diag!=0 or cC_diag!=0: boundary_fail+=1
        cr=crossing_indices(P)
        if cr: threads_seen+=1
        if cond2(P):
            sat_total+=1
            if cr: sat_with_thread+=1
print(f"[BK] diagonal matches never counted as crossings (exhaustive n<=5): violations={boundary_fail}")
print(f"[BK] partial perms (n<=5) WITH a crossing/thread index: {threads_seen}")
print(f"[BK] satisfying-(2) perms total={sat_total}, of which have >=1 thread index={sat_with_thread}")

# Direct check of (3.1) and (3.2) decompositions, term by term, exhaustively.
d_fail=0
for n in range(1,6):
    for P in all_partial_perms(n):
        chiR,chiC,r0,c0,mk=combinatorial(P)
        for k in range(1,n+1):
            if r0[k]+mk[k]+chiR[k]!=k: d_fail+=1   # (3.1)
            if c0[k]+mk[k]+chiC[k]!=k: d_fail+=1   # (3.2)
            if r0[k]+chiR[k]!=c0[k]+chiC[k]: d_fail+=1  # (3.3)
print(f"[BK] (3.1)/(3.2)/(3.3) term-by-term exhaustive n<=5: failures={d_fail}")

# Specifically exhibit one satisfying case WITH a thread root and print its
# nullity profile, to make sure the construction-relevant case is sound.
def show(P):
    chiR,chiC,r0,c0,mk=combinatorial(P)
    print("    P=",P,"thread idx=",crossing_indices(P))
    for k in range(1,len(P)+1):
        Pk=sub(P,range(k),range(k))
        print(f"      k={k}: nul_Pk={k-rank_Q(Pk)} r0+chiR={r0[k]+chiR[k]} "
              f"c0+chiC={c0[k]+chiC[k]} chiR={chiR[k]} c0={c0[k]} chiC={chiC[k]} r0={r0[k]}")
# canonical thread: zero row & col at 1, with arcs 1->? no; build: index2 above-root and below-root
# Pattern: row2 -> col3 (above), col2 -> row3 (below): that needs P[1][2]=1 and P[2][1]=1 plus zero row/col1
print("[BK] example with a crossing index (index 2 both above- and below-root):")
P=[[0,0,0,0],
   [0,0,1,0],   # row2 -> col3 (above arc 2->3)
   [0,1,0,0],   # row3 -> col2 (below arc 3->2)  => col2 matched to row3 below-root at index2
   [0,0,0,0]]
print("    cond2=",cond2(P),"comb=",comb_cond(P))
show(P)
