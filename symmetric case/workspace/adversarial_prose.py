"""
Check the two prose/algebra steps that are easy to state loosely:

(P-2.4) Identity B[:,1:k] = M A[:,1:k] N_k.
   The proof writes B[:,1:k] = M A N[:,1:k] and then uses N[:,1:k]=(N_k ; 0).
   The SUBTLE point: M A N[:,1:k] = M A (N_k ; 0) does NOT obviously equal
   M A[:,1:k] N_k unless (A N[:,1:k]) only involves the first k columns of A.
   Indeed A * (N_k ; 0) = A[:,1:k] * N_k  because the bottom block of N[:,1:k]
   is zero so only the first k columns of A multiply N_k. Verify EXACTLY.

(P-2.5) symmetric: B[1:k,:] = M_k A[1:k,:] N, via M[1:k,:]=(M_k|0):
   (M_k|0) * A = M_k * A[1:k,:]. Verify EXACTLY.

Done over Q with exact Fractions on random triangular M,N and random A,
PLUS a check that the rank/nullity invariance is exact (not float).
"""
from fractions import Fraction
import random
random.seed(2024)

def sub(M,rows,cols): return [[M[i][j] for j in cols] for i in rows]
def matmul(A,B):
    n,m,p=len(A),len(B),len(B[0])
    out=[[Fraction(0)]*p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            a=A[i][k]
            if a==0: continue
            for j in range(p):
                out[i][j]+=a*B[k][j]
    return out
def rank_Q(M):
    if not M: return 0
    A=[[Fraction(x) for x in r] for r in M]; nr=len(A); nc=len(A[0]); pr=0; rk=0
    for c in range(nc):
        piv=None
        for r in range(pr,nr):
            if A[r][c]!=0: piv=r;break
        if piv is None: continue
        A[pr],A[piv]=A[piv],A[pr]; pv=A[pr][c]
        for r in range(nr):
            if r!=pr and A[r][c]!=0:
                f=A[r][c]/pv; A[r]=[A[r][j]-f*A[pr][j] for j in range(nc)]
        pr+=1; rk+=1
    return rk
def rand_lower(n):
    return [[Fraction(random.randint(-3,3)) if j<i else (Fraction(random.choice([1,-1,2,-2])) if j==i else Fraction(0)) for j in range(n)] for i in range(n)]
def rand_upper(n):
    return [[Fraction(random.randint(-3,3)) if j>i else (Fraction(random.choice([1,-1,2,-2])) if j==i else Fraction(0)) for j in range(n)] for i in range(n)]

fail24=0; fail25=0; failnul=0
for _ in range(4000):
    n=random.randint(1,7)
    A=[[Fraction(random.randint(-3,3)) for _ in range(n)] for _ in range(n)]
    M=rand_lower(n); N=rand_upper(n)
    B=matmul(matmul(M,A),N)
    for k in range(1,n+1):
        rk=list(range(k)); ck=list(range(k)); allx=list(range(n))
        Nk=sub(N,ck,ck); Mk=sub(M,ck,ck)
        # P-2.4 : two ways of writing B[:,1:k]
        way1=sub(B,allx,ck)                          # actual leading cols of B
        way2=matmul(matmul(M,sub(A,allx,ck)),Nk)     # M A[:,1:k] N_k
        # also the intermediate identity A*(N[:,1:k]) == A[:,1:k]*N_k
        Ncolk=sub(N,allx,ck)                         # n x k, bottom block zero
        midL=matmul(A,Ncolk)
        midR=matmul(sub(A,allx,ck),Nk)
        if way1!=way2 or midL!=midR: fail24+=1
        # P-2.5
        way1b=sub(B,rk,allx)
        way2b=matmul(matmul(Mk,sub(A,rk,allx)),N)
        Mrowk=sub(M,rk,allx)                         # k x n, right block zero
        midL2=matmul(Mrowk,A)
        midR2=matmul(Mk,sub(A,rk,allx))
        if way1b!=way2b or midL2!=midR2: fail25+=1
        # exact nullity invariance
        Ak=sub(A,rk,ck); Bk=sub(B,rk,ck)
        Acol=sub(A,allx,ck); Bcol=sub(B,allx,ck)
        Arow=sub(A,rk,allx); Brow=sub(B,rk,allx)
        if (k-rank_Q(Ak))!=(k-rank_Q(Bk)): failnul+=1
        if (k-rank_Q(Acol))!=(k-rank_Q(Bcol)): failnul+=1
        if (k-rank_Q(Arow))!=(k-rank_Q(Brow)): failnul+=1
print(f"[P-2.4] B[:,1:k]=M A[:,1:k] N_k and A*(N[:,1:k])=A[:,1:k]*N_k exactly: failures={fail24}")
print(f"[P-2.5] B[1:k,:]=M_k A[1:k,:] N and (M[1:k,:])*A=M_k*A[1:k,:] exactly: failures={fail25}")
print(f"[NUL] exact nullity invariance of all 3 quantities under B=MAN: failures={failnul}")
