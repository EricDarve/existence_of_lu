# A rank/nullity criterion for unpivoted diagonal-$D$ $LDL^T$ factorization

This note records a field-general version of the criterion.  The matrix need not be real.  It is enough that the entries lie in a commutative field $F$; all ranks and nullities below are taken over $F$.

The result is **not** the usual positive-definite Cholesky theorem, and it is also not the standard pivoted block-$LDL^T$ factorization.  Here the factorization is unpivoted and the middle factor is truly diagonal.  The lower triangular factor is allowed to be singular.

Let
$$
A=A^T\in F^{n\times n},
$$
and define
$$
A_k := A(1\!:\!k,1\!:\!k),
\qquad
C_k := A(:,1\!:\!k).
$$
Here $C_k$ is an $n\times k$ matrix, and $\operatorname{nullity}(C_k)$ means the dimension of its right nullspace, i.e.
$$
\operatorname{nullity}(C_k)=k-\operatorname{rank}(C_k).
$$

## The symmetric criterion

For $A=A^T\in F^{n\times n}$, where $F$ is a commutative field, the following are equivalent:

1. There exist a lower triangular matrix $L\in F^{n\times n}$ and a diagonal matrix $D\in F^{n\times n}$ such that
   $$
   A = LDL^T,
   $$
   with zeros allowed on the diagonal of $L$.

2. For every $k=1,\dots,n$,
   $$
   \operatorname{nullity}(A_k)\le 2\,\operatorname{nullity}(C_k).
   $$

Equivalently,
$$
\operatorname{rank}(A_k)+k \ge 2\,\operatorname{rank}(C_k)
\qquad (k=1,\dots,n).
$$

The statement should not be extended naively from fields to general commutative rings: the proof uses ranks, nullities, inverses of nonzero field elements, and Sylvester-type rank inequalities.

## Proof

### Necessity

Assume
$$
A=LDL^T,
$$
where $L$ is lower triangular and $D$ is diagonal.  Partition
$$
L=
\begin{bmatrix}
L_{11} & 0\\
L_{21} & L_{22}
\end{bmatrix},
\qquad
D=
\begin{bmatrix}
D_1 & 0\\
0 & D_2
\end{bmatrix},
$$
with $L_{11},D_1$ of size $k$.

Then
$$
A_k = L_{11}D_1L_{11}^T,
\qquad
C_k =
\begin{bmatrix}
L_{11}\\
L_{21}
\end{bmatrix}
D_1L_{11}^T.
$$
Hence
$$
\operatorname{rank}(C_k)\le \operatorname{rank}(L_{11}),
\qquad
\operatorname{rank}(C_k)\le \operatorname{rank}(D_1L_{11}^T).
$$
Applying Sylvester's rank inequality to
$$
A_k=L_{11}(D_1L_{11}^T)
$$
gives
$$
\operatorname{rank}(A_k)
\ge
\operatorname{rank}(L_{11})
+
\operatorname{rank}(D_1L_{11}^T)
-k
\ge
2\operatorname{rank}(C_k)-k.
$$
Therefore
$$
\operatorname{rank}(A_k)+k\ge 2\operatorname{rank}(C_k),
$$
equivalently
$$
\operatorname{nullity}(A_k)\le 2\,\operatorname{nullity}(C_k).
$$
This argument is field-independent.

### Sufficiency

The sufficiency proof has three steps.  The real-field proof can use a signed matching matrix and a division by $2$ in one local congruence.  Over an arbitrary field, and especially in characteristic $2$, it is better to use a **weighted matching matrix**.  This avoids signs, square roots, and division by $2$.

#### Weighted matching matrices

Call a symmetric matrix $P\in F^{n\times n}$ a weighted matching matrix if its indices are partitioned into blocks of the following types:

- isolated zero indices;
- nonzero diagonal singletons $[\gamma]$, with $\gamma\in F^\times$;
- ordered two-index blocks $(i,j)$, $i<j$, of the form
  $$
  \begin{bmatrix}
  0 & \alpha\\
  \alpha & \beta
  \end{bmatrix},
  \qquad
  \alpha\in F^\times,
  \quad
  \beta\in F.
  $$

All entries outside these blocks are zero.  Thus the earlier index of a two-index block has zero diagonal entry, the off-diagonal weight is nonzero, and the later index may carry an arbitrary diagonal weight.  The block is nonsingular because its determinant is $-\alpha^2\ne0$.

#### Step 1: lower-triangular congruence reduction to a weighted matching matrix

I claim that every symmetric matrix $A\in F^{n\times n}$ can be reduced by invertible lower-triangular congruence to a weighted matching matrix.  More precisely, one can construct an invertible lower triangular matrix $M$ and a weighted matching matrix $P$ such that
$$
A=MPM^T.
$$

The proof is by induction on the leading dimension.  Suppose
$$
A_{k-1}=M_{k-1}P_{k-1}M_{k-1}^T,
$$
where $M_{k-1}$ is invertible lower triangular and $P_{k-1}$ is a weighted matching matrix.  Write
$$
A_k=
\begin{bmatrix}
A_{k-1} & a\\
a^T & b
\end{bmatrix},
\qquad
c:=M_{k-1}^{-1}a.
$$
Then
$$
A_k=
\begin{bmatrix}
M_{k-1} & 0\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
P_{k-1} & c\\
c^T & b
\end{bmatrix}
\begin{bmatrix}
M_{k-1}^T & 0\\
0 & 1
\end{bmatrix}.
$$

On each nonzero block of $P_{k-1}$, the matrix $P_{k-1}$ is invertible.  Choose $\xi$ blockwise so that $P_{k-1}\xi$ agrees with $c$ on all nonzero blocks, and choose $\xi=0$ on isolated zero indices.  Then
$$
e:=c-P_{k-1}\xi
$$
is supported only on isolated zero indices of $P_{k-1}$, and $\xi^Te=0$.  Let
$$
f:=b-\xi^TP_{k-1}\xi.
$$
Then
$$
\begin{bmatrix}
P_{k-1} & c\\
c^T & b
\end{bmatrix}
=
\begin{bmatrix}
I & 0\\
\xi^T & 1
\end{bmatrix}
\begin{bmatrix}
P_{k-1} & e\\
e^T & f
\end{bmatrix}
\begin{bmatrix}
I & \xi\\
0 & 1
\end{bmatrix}.
$$

If $e=0$, the middle block is just $P_{k-1}\oplus[f]$.  If $f=0$, this appends an isolated zero; if $f\ne0$, it appends a nonzero diagonal singleton.

If $e\ne0$, let $i$ be the first nonzero component of $e$, and set
$$
\alpha:=e_i\ne0.
$$
Since $e$ is supported only on isolated zero indices, the index $i$ is isolated in $P_{k-1}$.  Define
$$
R:=I+(\alpha^{-1}e-u_i)u_i^T,
$$
where $u_i$ is the $i$th standard basis vector.  Because $i$ is the first nonzero component of $e$, the matrix $R$ is lower triangular and invertible.  Also $R P_{k-1}R^T=P_{k-1}$, because row and column $i$ of $P_{k-1}$ are zero.  Therefore
$$
\begin{bmatrix}
P_{k-1} & e\\
e^T & f
\end{bmatrix}
=
\begin{bmatrix}
R & 0\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
P_{k-1} & \alpha u_i\\
\alpha u_i^T & f
\end{bmatrix}
\begin{bmatrix}
R^T & 0\\
0 & 1
\end{bmatrix}.
$$
The middle matrix is again a weighted matching matrix: it connects the previously isolated index $i$ to the new index $k$ with off-diagonal weight $\alpha$ and later diagonal weight $f$.

Thus, by induction,
$$
A=MPM^T
$$
for some invertible lower triangular $M$ and some weighted matching matrix $P$.

#### Step 2: the nullity condition transfers from $A$ to $P$

If
$$
B=TAT^T
$$
with $T$ invertible lower triangular, then for every $k$,
$$
B_k=T_kA_kT_k^T,
\qquad
B(:,1\!:\!k)=T\,A(:,1\!:\!k)\,T_k^T,
$$
where $T_k$ is the leading $k\times k$ block of $T$.  Since $T$ and $T_k$ are invertible,
$$
\operatorname{nullity}(B_k)=\operatorname{nullity}(A_k),
\qquad
\operatorname{nullity}(B(:,1\!:\!k))=\operatorname{nullity}(A(:,1\!:\!k)).
$$
So $A$ satisfies
$$
\operatorname{nullity}(A_k)\le 2\,\operatorname{nullity}(C_k)
$$
for every $k$ if and only if $P$ satisfies the corresponding condition.

#### Step 3: the weighted matching condition gives an explicit diagonal-$D$ factorization

Let $P$ be a weighted matching matrix.  Define:

- $\zeta_k$ = the number of isolated zero indices among $\{1,\dots,k\}$;
- $\chi_k$ = the number of two-index blocks $(i,j)$ with $i\le k<j$.

Then:

- in $P_k$, each isolated zero index contributes one null direction, and each crossing block $(i,j)$ with $i\le k<j$ contributes one additional null direction, so
  $$
  \operatorname{nullity}(P_k)=\zeta_k+\chi_k;
  $$
- in $P(:,1\!:\!k)$, the only zero columns are the isolated-zero columns, while each crossing block contributes a nonzero independent column, so
  $$
  \operatorname{nullity}(P(:,1\!:\!k))=\zeta_k.
  $$

Therefore the hypothesis becomes exactly
$$
\chi_k\le \zeta_k
\qquad (k=1,\dots,n). \tag{$*$}
$$

Now scan the indices from left to right.

Each isolated zero creates one free **thread**.  When a two-index block $(i,j)$ starts at $i$, assign it to any free thread.  When it ends at $j$, that thread becomes free again.  At step $k$, the busy threads are exactly the crossing blocks counted by $\chi_k$, while the threads created so far are counted by $\zeta_k$.  Hence condition $(*)$ guarantees that a free thread is always available.  Thus the two-index blocks split into disjoint threads of the form
$$
r<i_1<j_1<i_2<j_2<\cdots<i_m<j_m,
$$
where $r$ is an isolated zero index that roots the thread.

Consider one such thread.  Write the weights of its two-index blocks as
$$
P_{i_qj_q}=P_{j_qi_q}=\alpha_q\ne0,
\qquad
P_{j_qj_q}=\beta_q,
\qquad q=1,\dots,m.
$$
The diagonal entries at the start indices $i_q$ are zero.

Set
$$
s_{m+1}:=0,
\qquad
\delta_{m+1}:=1.
$$
Proceed backward for $q=m,m-1,\dots,1$.  Choose $v_q\in F$ so that
$$
\delta_q:=\delta_{q+1}v_q^2-\beta_q
$$
is nonzero.  This is always possible: if $\beta_q\ne0$, take $v_q=0$; if $\beta_q=0$, take $v_q=1$.  Then define
$$
s_q:=-\frac{\alpha_q}{\delta_q}e_{i_q}
+
\frac{\delta_{q+1}v_q}{\delta_q}s_{q+1}.
$$

Now define the thread columns and diagonal weights by
$$
\ell_r:=s_1,
\qquad
 d_r:=\delta_1,
$$
 and, for $q=1,\dots,m$,
$$
\ell_{i_q}:=s_q+e_{j_q},
\qquad
 d_{i_q}:=-\delta_q,
$$
$$
\ell_{j_q}:=s_{q+1}+v_qe_{j_q},
\qquad
 d_{j_q}:=\delta_{q+1}.
$$
Because
$$
r<i_1<j_1<i_2<j_2<\cdots<i_m<j_m,
$$
each $\ell_t$ is supported only in rows $\ge t$, so these are valid columns of a lower triangular matrix.  The identities
$$
-\delta_qs_q+\delta_{q+1}v_qs_{q+1}=\alpha_qe_{i_q},
\qquad
-\delta_q+\delta_{q+1}v_q^2=\beta_q
$$
show that
$$
\begin{aligned}
&\delta_qs_qs_q^T
-\delta_q(s_q+e_{j_q})(s_q+e_{j_q})^T
+\delta_{q+1}(s_{q+1}+v_qe_{j_q})(s_{q+1}+v_qe_{j_q})^T
-\delta_{q+1}s_{q+1}s_{q+1}^T \\
&\qquad =
\alpha_q(e_{i_q}e_{j_q}^T+e_{j_q}e_{i_q}^T)
+
\beta_qe_{j_q}e_{j_q}^T.
\end{aligned}
$$
Summing over $q$ telescopes the $\delta_qs_qs_q^T$ terms and gives exactly the contribution of this thread.

Nonzero diagonal singletons $[\gamma]$ are factored trivially as
$$
\gamma e_te_t^T.
$$
Isolated zero indices that do not root a nonempty thread contribute zero columns.  Distinct threads use disjoint index sets, so their contributions do not interact.  Therefore
$$
P=SES^T
$$
for some lower triangular $S$ and diagonal $E$ over $F$.

Finally,
$$
A=MPM^T=MSES^TM^T=(MS)E(MS)^T,
$$
and $MS$ is lower triangular.  Hence $A$ has an unpivoted diagonal-$D$ $LDL^T$ factorization.

This proves sufficiency over any commutative field.

## Check on
$$
A=
\begin{bmatrix}
0&0&0\\
0&0&1\\
0&1&0
\end{bmatrix}.
$$

This matrix is already a weighted matching matrix: index $1$ is an isolated zero, and $(2,3)$ is one two-index block with
$$
\alpha_1=1,
\qquad
\beta_1=0.
$$
Thus
$$
\zeta_1=\zeta_2=\zeta_3=1,
\qquad
\chi_1=0,
\quad
\chi_2=1,
\quad
\chi_3=0.
$$
So $\chi_k\le\zeta_k$ for every $k$, exactly as required.

The field-general thread construction gives one valid factorization as follows.  Choose
$$
\delta_2=1,
\qquad
v_1=1,
\qquad
\delta_1=1,
\qquad
s_2=0,
\qquad
s_1=-e_2.
$$
Then
$$
L=
\begin{bmatrix}
0&0&0\\
-1&-1&0\\
0&1&1
\end{bmatrix},
\qquad
D=\operatorname{diag}(1,-1,1),
$$
and indeed
$$
LDL^T=
\begin{bmatrix}
0&0&0\\
0&0&1\\
0&1&0
\end{bmatrix}.
$$
In characteristic $2$, the same displayed factorization is interpreted with $-1=1$, and it remains valid.

The example shows why allowing a zero diagonal entry in $L$ is essential: the leading zero index is the root of the thread that realizes the later two-index block $(2,3)$.

## References

These references are related background for no-pivot factorization and symmetric elimination.  The proof above is written directly over a field and does not rely on a real-only signed reduction.

- Pavel Okunev and Charles R. Johnson, *Necessary and Sufficient Conditions for Existence of the LU Factorization of an Arbitrary Matrix* (2005).  
  <https://arxiv.org/abs/math/0506382>

- Eric Darve, *Necessary and Sufficient Conditions for the Existence of an LU Factorization for General Rank Deficient Matrices* (2026).  
  <https://arxiv.org/abs/2601.07791>

- Froilán M. Dopico, Charles R. Johnson, and Juan M. Molera, *Multiple LU factorizations of a singular matrix*, *Linear Algebra and its Applications* **419** (2006), 24–36.  
  <https://doi.org/10.1016/j.laa.2006.03.043>

- Cony M. Lau and Thomas L. Markham, *LU factorizations*, *Czechoslovak Mathematical Journal* **29**(4) (1979), 546–550.  
  <https://doi.org/10.21136/CMJ.1979.101635>

- Paul Van Dooren and Gilbert Strang, *Symmetric elimination without pivoting*, *Linear Algebra and its Applications* **452** (2014), 40–45.  
  <https://doi.org/10.1016/j.laa.2014.03.026>


## Unsymmetric criterion — proof of sufficiency (matching matrices and threads)

We now drop symmetry. Let $F$ be a commutative field and let $A\in F^{n\times n}$ be **arbitrary** (no longer assumed equal to $A^T$). The factors are now a lower triangular $L$ and an upper triangular $U$, each allowed to carry zeros on its diagonal. As in the symmetric case write
$$
A[1\!:\!k,1\!:\!k]\quad(k\times k),\qquad
A[:,1\!:\!k]\quad(n\times k),\qquad
A[1\!:\!k,:]\quad(k\times n),
$$
and let $\operatorname{nullity}$ of a rectangular matrix denote the dimension of its **right** nullspace, so that for an $m\times p$ matrix $X$, $\operatorname{nullity}(X)=p-\operatorname{rank}(X)$. In particular
$$
\operatorname{nullity}(A[:,1\!:\!k])=k-\operatorname{rank}(A[:,1\!:\!k]),
\qquad
\operatorname{nullity}(A[1\!:\!k,:]^T)=k-\operatorname{rank}(A[1\!:\!k,:]).
$$

The criterion is the unsymmetric analogue of the symmetric one — it is **Property 2.1** of Darve (2026). Note two differences from the displayed "Unsymmetric criterion" statement above: the matrix is **not** assumed symmetric, and the last term is $\operatorname{nullity}(A[1\!:\!k,:]^T)$ (the transpose), i.e. $k-\operatorname{rank}(A[1\!:\!k,:])$, rather than $\operatorname{nullity}(A[1\!:\!k,:])$.

**Unsymmetric criterion (corrected).** For $A\in F^{n\times n}$ over a commutative field $F$, the following are equivalent:

1. There exist a lower triangular $L\in F^{n\times n}$ and an upper triangular $U\in F^{n\times n}$, with zeros allowed on **both** diagonals, such that
   $$
   A = LU.
   $$

2. For every $k=1,\dots,n$,
   $$
   \operatorname{nullity}(A[1\!:\!k,1\!:\!k])\ \le\
   \operatorname{nullity}(A[:,1\!:\!k]) + \operatorname{nullity}(A[1\!:\!k,:]^T).
   \tag{2}
   $$

The symmetric criterion $\operatorname{nullity}(A_k)\le 2\operatorname{nullity}(C_k)$ is the specialization of $(2)$ to $A=A^T$: there $A[:,1\!:\!k]$ and $A[1\!:\!k,:]^T$ have equal nullity, and the two terms on the right collapse into the factor $2$.

As in the symmetric case the statement is field-general but not ring-general: the proof uses ranks, nullities, inverses of nonzero field elements, and Sylvester's rank inequality.

### Necessity

Assume $A=LU$ with $L$ lower triangular and $U$ upper triangular. Partition at $k$,
$$
L=\begin{bmatrix}L_{11}&0\\ L_{21}&L_{22}\end{bmatrix},
\qquad
U=\begin{bmatrix}U_{11}&U_{12}\\ 0&U_{22}\end{bmatrix},
$$
with $L_{11},U_{11}$ of size $k$. Reading off the leading block, the first $k$ columns, and the first $k$ rows of $A=LU$ — using that $L$ lower triangular gives $L[1\!:\!k,:]=\begin{bmatrix}L_{11}&0\end{bmatrix}$ and $U$ upper triangular gives $U[:,1\!:\!k]=\begin{bmatrix}U_{11}\\ 0\end{bmatrix}$ — yields
$$
A[1\!:\!k,1\!:\!k]=L_{11}U_{11},
\qquad
A[:,1\!:\!k]=\begin{bmatrix}L_{11}\\ L_{21}\end{bmatrix}U_{11},
\qquad
A[1\!:\!k,:]=L_{11}\begin{bmatrix}U_{11}&U_{12}\end{bmatrix}.
$$
Hence
$$
\operatorname{rank}(A[:,1\!:\!k])\le \operatorname{rank}(U_{11}),
\qquad
\operatorname{rank}(A[1\!:\!k,:])\le \operatorname{rank}(L_{11}).
$$
Applying Sylvester's rank inequality to $A[1\!:\!k,1\!:\!k]=L_{11}U_{11}$,
$$
\operatorname{rank}(A[1\!:\!k,1\!:\!k])
\ \ge\
\operatorname{rank}(L_{11})+\operatorname{rank}(U_{11})-k
\ \ge\
\operatorname{rank}(A[1\!:\!k,:])+\operatorname{rank}(A[:,1\!:\!k])-k.
$$
Rearranging and using $\operatorname{nullity}(X)=(\#\text{cols of }X)-\operatorname{rank}(X)$ on each of the three $k$-column matrices $A[1\!:\!k,1\!:\!k]$, $A[:,1\!:\!k]$, $A[1\!:\!k,:]^T$ gives
$$
\operatorname{nullity}(A[1\!:\!k,1\!:\!k])\ \le\
\operatorname{nullity}(A[:,1\!:\!k]) + \operatorname{nullity}(A[1\!:\!k,:]^T).
$$
This argument is field-independent. $\blacksquare$

### Sufficiency

The sufficiency proof has three steps that mirror the symmetric proof. There the two outer factors are forced to be transposes of one another (the congruence $A=MPM^T$) and the middle factor is a *symmetric* weighted matching matrix; here the left and right reductions decouple into an independent lower factor $M$ and upper factor $N$, and the middle factor is a **partial permutation matrix**.

By a **partial permutation matrix** $\Pi\in F^{n\times n}$ we mean a $0/1$ matrix with at most one $1$ in each row and at most one $1$ in each column. Equivalently there is an injective partial map $\sigma$ from a subset of row indices to column indices with $\Pi_{i,\sigma(i)}=1$ and all other entries zero. We call the $1$-positions the **matched pairs** $(i,j)$ (meaning $\Pi_{ij}=1$), a row carrying no $1$ a **zero row**, and a column carrying no $1$ a **zero column**.

#### Step 1: lower- and upper-triangular reduction to a partial permutation matrix

I claim that every $A\in F^{n\times n}$ can be written as
$$
A=M\,\Pi\,N,
$$
where $M$ is invertible lower triangular, $N$ is invertible upper triangular, and $\Pi$ is a partial permutation matrix. This is the field-general LPU (generalized Bruhat) decomposition. The proof is by induction on the leading dimension $k$, mirroring the symmetric Step 1: there the two outer factors are forced to be transposes ($MPM^T$), whereas here the left and right reductions are decoupled into a lower factor $M$ and an upper factor $N$.

Throughout write $A_k:=A[1\!:\!k,1\!:\!k]$.

**Inductive hypothesis.** Suppose
$$
A_{k-1}=M_{k-1}\,\Pi_{k-1}\,N_{k-1},
$$
where $M_{k-1}\in F^{(k-1)\times(k-1)}$ is invertible lower triangular, $N_{k-1}$ is invertible upper triangular, and $\Pi_{k-1}$ is a partial permutation matrix.

**Border the matrix.** Write
$$
A_k=
\begin{bmatrix}
A_{k-1} & a\\
c^T & b
\end{bmatrix},
\qquad
a,c\in F^{k-1},\;\; b\in F,
$$
and pass to **reduced coordinates** by stripping the known outer factors. Put
$$
\widehat a:=M_{k-1}^{-1}a,
\qquad
\widehat c^{\,T}:=c^T N_{k-1}^{-1}
\quad(\text{i.e. } \widehat c:=N_{k-1}^{-T}c).
$$
Then
$$
A_k=
\begin{bmatrix}
M_{k-1} & 0\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\Pi_{k-1} & \widehat a\\
\widehat c^{\,T} & b
\end{bmatrix}
\begin{bmatrix}
N_{k-1} & 0\\
0 & 1
\end{bmatrix}.
\tag{1.1}
$$
The two outer factors in $(1.1)$ are invertible lower and upper triangular. It remains to reduce the bordered middle matrix
$$
B:=
\begin{bmatrix}
\Pi_{k-1} & \widehat a\\
\widehat c^{\,T} & b
\end{bmatrix}
$$
to a partial permutation using only **further** lower-triangular row operations and upper-triangular column operations supported on $\{1,\dots,k\}$, absorbing them into the outer factors. We use two families of elementary operations, both of which preserve the leading block $\Pi_{k-1}$:

- **(Upper column ops.)** Adding a multiple of an earlier column $j<k$ to the last column $k$ is right multiplication by $I+\lambda\,e_je_k^T$, upper triangular; it is absorbed into $N$. Adding a multiple of an earlier column $j$ to a column $j'$ with $j<j'<k$ is likewise upper triangular.
- **(Lower row ops.)** Adding a multiple of an earlier row $i<k$ to the last row $k$ is left multiplication by $I+\lambda\,e_ke_i^T$, lower triangular; it is absorbed into $M$. Adding a multiple of an earlier row $i$ to a row $i'$ with $i<i'<k$ is likewise lower triangular. (Scaling the last row or column by a nonzero field element is diagonal, hence both lower and upper triangular, and is absorbed accordingly.)

We reduce $B$ in three stages. The working matrix is the **full** $n\times n$ matrix; we only require $\Pi_k$ to be a partial permutation on the leading $(k+1)\times(k+1)$ block at stage $k$, because entries created in columns $>k$ or rows $>k$ lie outside that block and are cleared at later stages.

**Stage 1: clear the new column over matched rows.** For each matched pair $(i,j)$ of $\Pi_{k-1}$ (so $(\Pi_{k-1})_{ij}=1$, $i,j<k$), the entry $\widehat a_i$ lies in a matched row. Subtract $\widehat a_i$ times column $j$ from the last column:
$$
\text{col}_k \;\leftarrow\; \text{col}_k-\widehat a_i\,\text{col}_j .
$$
Within the leading block, column $j$ of $B$ has its only nonzero entry, a $1$, at row $i$, so this replaces $\widehat a_i$ by $0$ and changes no other entry of the leading block. Doing this for all matched pairs, the residual new column
$$
e:=\bigl(\text{updated }\widehat a\bigr)
$$
is supported, within the leading block, only on **zero rows** of $\Pi_{k-1}$. Each step is an upper-triangular column op ($j<k$), absorbed into $N$.

**Stage 2: clear the new row over matched columns.** Symmetrically, for each matched pair $(i,j)$ the entry $\widehat c_{\,j}$ lies in a matched column. Subtract $\widehat c_{\,j}$ times row $i$ from the last row:
$$
\text{row}_k\;\leftarrow\;\text{row}_k-\widehat c_{\,j}\,\text{row}_i .
$$
Within the leading block, row $i$ of $\Pi_{k-1}$ has its only nonzero entry at column $j$, so this clears $\widehat c_{\,j}$ and disturbs no other leading-block entry. The residual new row
$$
g:=\bigl(\text{updated }\widehat c\,\bigr)
$$
is then supported, within the leading block, only on **zero columns** of $\Pi_{k-1}$. Each step is a lower-triangular row op ($i<k$), absorbed into $M$.

After Stages 1–2 the leading-block portion of the bordered matrix has the form
$$
\begin{bmatrix}
\Pi_{k-1} & e\\
g^{T} & b'
\end{bmatrix},
\qquad
\operatorname{supp}(e)\subseteq\{\text{zero rows}\},\quad
\operatorname{supp}(g)\subseteq\{\text{zero columns}\},
$$
for some corner $b'\in F$.

**Stage 3: normalize the residual column, the residual row, and the corner.**

*Column.* If $e\ne0$, let $i^\star$ be the **first** index with $e_{i^\star}\ne0$; by Stage 1, $i^\star$ is a zero row of $\Pi_{k-1}$. Scale the last column by $e_{i^\star}^{-1}$ so the entry at $(i^\star,k)$ becomes $1$ (a diagonal column scaling, absorbed into $N$). For each later zero-row index $i$ with $i^\star<i<k$ and residual entry still nonzero, eliminate it by
$$
\text{row}_i\;\leftarrow\;\text{row}_i-(\text{entry})\cdot\text{row}_{i^\star}.
$$
Row $i^\star$ is zero on columns $1,\dots,k-1$ (it is a zero row of $\Pi_{k-1}$) and equals $1$ at $(i^\star,k)$; hence this operation removes the entry at $(i,k)$ and the only other entries it can affect lie in columns $>k$, which are **outside the leading $(k+1)$ block**. Since $i^\star<i$ it is a lower-triangular row op, absorbed into $M$. We have created a new matched pair $(i^\star,k)$, consuming the formerly-zero row $i^\star$. (If $e=0$, no new column match is created.)

*Row.* Symmetrically, if $g\ne0$, let $j^\star$ be the first index with $g_{j^\star}\ne0$; it is a zero column of $\Pi_{k-1}$. Scale the last row by $g_{j^\star}^{-1}$ (diagonal, absorbed into $M$), then for each later zero-column index $j$ with $j^\star<j<k$ clear it by
$$
\text{col}_j\;\leftarrow\;\text{col}_j-(\text{entry})\cdot\text{col}_{j^\star},
$$
an upper-triangular column op ($j^\star<j$), absorbed into $N$. Column $j^\star$ is zero on rows $1,\dots,k-1$ and equals $1$ at $(k,j^\star)$, so the only entries these ops can disturb beyond the cleared $(k,j)$ lie in rows $>k$, outside the leading $(k+1)$ block; in particular they do not damage the column-match $(i^\star,k)$ created above, whose support lies in column $k\le k$ (these column ops touch only columns $j<k$). We have created a new matched pair $(k,j^\star)$, consuming the formerly-zero column $j^\star$. (If $g=0$, no new row match is created.)

*Corner.* Finally reconcile the $(k,k)$ entry, whose current value call $w$.

- If a column match $(i^\star,k)$ was created, clear the corner by $\text{row}_k\leftarrow\text{row}_k-w\,\text{row}_{i^\star}$. Row $i^\star$ is zero on columns $1,\dots,k-1$ and is $1$ at column $k$, so this sets $(k,k)\to0$ and changes row $k$ only in columns $>k$, which lie outside the leading $(k+1)$ block. It is a lower-triangular row op ($i^\star<k$), absorbed into $M$.
- Otherwise, if a row match $(k,j^\star)$ was created, clear the corner symmetrically by $\text{col}_k\leftarrow\text{col}_k-w\,\text{col}_{j^\star}$, an upper-triangular column op absorbed into $N$, which changes column $k$ only in rows $>k$.
- If neither match was created, then $e=0$ and $g=0$, and the leading block is $\Pi_{k-1}\oplus[\,w\,]$. If $w\ne0$, scale the last column by $w^{-1}$ (absorbed into $N$) to put a $1$ at $(k,k)$, creating the matched pair $(k,k)$; if $w=0$, index $k$ becomes an isolated zero row and zero column.

In every case the leading $(k+1)\times(k+1)$ block has been reduced, by lower-triangular row operations and upper-triangular column operations, to a partial permutation $\Pi_k$: Stages 1–2 leave $\Pi_{k-1}$ intact, and Stage 3 places at most one new $1$ in the last column (at a previously zero row $i^\star$), at most one new $1$ in the last row (at a previously zero column $j^\star$), or at most one $1$ at the corner — never violating the at-most-one-per-line condition. Spurious entries created in rows $>k$ or columns $>k$ lie outside this block and are removed by Stages 1–2 of the subsequent borders.

**Conclusion of the induction.** Folding the elementary operations of Stages 1–3 into the outer factors of $(1.1)$ produces an invertible lower triangular $M_k$ and an invertible upper triangular $N_k$ with $A_k=M_k\,\Pi_k\,N_k$, $\Pi_k$ a partial permutation. The base case $k=1$ is immediate: $A_1=[a_{11}]=[1][1][a_{11}]$ if $a_{11}\ne0$ (matched pair $(1,1)$) and $[1][0][1]$ if $a_{11}=0$ (isolated zero). Taking $k=n$ gives
$$
A=M\,\Pi\,N
$$
with $M$ invertible lower triangular, $N$ invertible upper triangular, $\Pi$ a partial permutation. The construction uses only field addition, multiplication, and inversion of the nonzero pivots $e_{i^\star},g_{j^\star},w$ — no signs, square roots, or division by $2$ — and is therefore valid over an arbitrary commutative field, characteristic $2$ included. $\blacksquare$

#### Step 2: the condition transfers from $A$ to $\Pi$

Let $M\in F^{n\times n}$ be invertible lower triangular, $N\in F^{n\times n}$ invertible upper triangular, and set $B=MAN$. Write $M_k:=M[1\!:\!k,1\!:\!k]$, $N_k:=N[1\!:\!k,1\!:\!k]$. Because $M$ is lower triangular and $N$ upper triangular,
$$
M[1\!:\!k,:]=\begin{bmatrix}M_k & 0\end{bmatrix},
\qquad
N[:,1\!:\!k]=\begin{bmatrix}N_k\\ 0\end{bmatrix}.
\tag{2.1}
$$
Moreover $M$ invertible lower triangular has all diagonal entries nonzero (its determinant is their product), so the triangular leading block $M_k$ — carrying those same first $k$ diagonal entries — is invertible; likewise $N_k$.

Using $(2.1)$, three block identities follow:

- *Leading block.* $B[1\!:\!k,1\!:\!k]=M[1\!:\!k,:]\,A\,N[:,1\!:\!k]$; the left factor kills rows of $A$ beyond $k$ and the right factor kills columns beyond $k$, so
  $$
  B[1\!:\!k,1\!:\!k]=M_k\,A[1\!:\!k,1\!:\!k]\,N_k.
  \tag{2.2}
  $$
- *Leading columns.* $B[:,1\!:\!k]=MA\,N[:,1\!:\!k]$; since the bottom block of $N[:,1\!:\!k]$ is zero, $A\,N[:,1\!:\!k]=A[:,1\!:\!k]\,N_k$, whence
  $$
  B[:,1\!:\!k]=M\,A[:,1\!:\!k]\,N_k.
  \tag{2.3}
  $$
- *Leading rows.* $B[1\!:\!k,:]=M[1\!:\!k,:]\,AN$; since the right block of $M[1\!:\!k,:]$ is zero, $M[1\!:\!k,:]\,A=M_k\,A[1\!:\!k,:]$, whence
  $$
  B[1\!:\!k,:]=M_k\,A[1\!:\!k,:]\,N.
  \tag{2.4}
  $$

**Nullity invariance.** Left or right multiplication by an invertible matrix preserves rank, and (for a fixed number of columns) right nullity. In $(2.2)$ both $M_k,N_k$ are invertible and both sides are $k\times k$, so $\operatorname{nullity}(B[1\!:\!k,1\!:\!k])=\operatorname{nullity}(A[1\!:\!k,1\!:\!k])$. In $(2.3)$ the left factor $M$ and right factor $N_k$ are invertible and both sides are $n\times k$, so $\operatorname{nullity}(B[:,1\!:\!k])=\operatorname{nullity}(A[:,1\!:\!k])$. Transposing $(2.4)$ gives $B[1\!:\!k,:]^T=N^T A[1\!:\!k,:]^T M_k^T$ with $N^T,M_k^T$ invertible and both sides $n\times k$, so $\operatorname{nullity}(B[1\!:\!k,:]^T)=\operatorname{nullity}(A[1\!:\!k,:]^T)$.

Thus each of the three nullities in $(2)$ is identical for $A$ and for $B=MAN$. Applying this with $\Pi=M^{-1}AN^{-1}$ from Step 1 ($M^{-1}$ invertible lower triangular, $N^{-1}$ invertible upper triangular):

> **$A$ satisfies $(2)$ if and only if its partial permutation $\Pi$ does.** $\blacksquare$

#### Step 3: the partial-permutation condition gives an explicit LU factorization

Let $\Pi$ be a partial permutation matrix with matched-pair set $\mathcal M=\{(i,j):\Pi_{ij}=1\}$. Each matched pair is exactly one of: a **diagonal pair** $(t,t)$; an **above-arc** $(i,j)$ with $i<j$; or a **below-arc** $(i,j)$ with $i>j$. For $k=1,\dots,n$ define
$$
\begin{aligned}
\chi^R_k &= \#\{(i,j)\in\mathcal M:\ i\le k<j\} &&\text{(above-arcs crossing the cut }k),\\
\chi^C_k &= \#\{(i,j)\in\mathcal M:\ j\le k<i\} &&\text{(below-arcs crossing the cut }k),\\
r^0_k &= \#\{\text{zero rows}\le k\},\qquad
c^0_k = \#\{\text{zero columns}\le k\},
\end{aligned}
$$
and let $\iota_k=\#\{(i,j)\in\mathcal M:\ i\le k\text{ and }j\le k\}$ count the matched pairs inside the leading $k\times k$ block.

**Lemma 3.1 (nullity formulas).** For every $k$,
$$
\operatorname{nullity}(\Pi[1\!:\!k,1\!:\!k])=r^0_k+\chi^R_k=c^0_k+\chi^C_k,
\tag{3.1}
$$
$$
\operatorname{nullity}(\Pi[:,1\!:\!k])=c^0_k,
\qquad
\operatorname{nullity}(\Pi[1\!:\!k,:]^T)=r^0_k.
\tag{3.2}
$$

*Proof.* The $n\times k$ matrix $\Pi[:,1\!:\!k]$ has, in each matched column $j\le k$, a single nonzero entry $e_i$; these are distinct standard basis vectors (distinct because $\Pi$ has at most one $1$ per row), hence independent, while zero-column indices $\le k$ give zero columns. Thus $\operatorname{rank}(\Pi[:,1\!:\!k])=k-c^0_k$, so its right nullity is $c^0_k$. Applying the same to $\Pi^T$ gives $\operatorname{rank}(\Pi[1\!:\!k,:])=k-r^0_k$, so $\operatorname{nullity}(\Pi[1\!:\!k,:]^T)=k-\operatorname{rank}(\Pi[1\!:\!k,:])=r^0_k$. This is $(3.2)$.

The leading block $\Pi[1\!:\!k,1\!:\!k]$ is itself a partial permutation whose $1$'s are the $\iota_k$ pairs inside the block, occupying distinct rows and columns; a $0/1$ matrix with at most one $1$ per row and column has rank equal to its number of $1$'s (its nonzero rows are distinct basis vectors), so $\operatorname{rank}(\Pi[1\!:\!k,1\!:\!k])=\iota_k$ and $\operatorname{nullity}=k-\iota_k$. Classify the $k$ row indices $\le k$: each is a zero row, matched inside the block ($j\le k$), or matched to a column $>k$ (an above-arc with $i\le k<j$); these classes are disjoint and exhaust $\{1,\dots,k\}$, so $k=\iota_k+r^0_k+\chi^R_k$, i.e. $k-\iota_k=r^0_k+\chi^R_k$. Classifying the $k$ column indices instead gives $k=\iota_k+c^0_k+\chi^C_k$, i.e. $k-\iota_k=c^0_k+\chi^C_k$. This proves $(3.1)$ and, as a by-product, the **bookkeeping identity**
$$
r^0_k+\chi^R_k=c^0_k+\chi^C_k.
\tag{3.3}
$$
$\blacksquare$

**Combinatorial form of (2).** Substituting $(3.1)$–$(3.2)$ into $(2)$ for $\Pi$ gives, via the first form of $(3.1)$, $r^0_k+\chi^R_k\le c^0_k+r^0_k$, i.e. $\chi^R_k\le c^0_k$; via the second form, $c^0_k+\chi^C_k\le c^0_k+r^0_k$, i.e. $\chi^C_k\le r^0_k$. By $(3.3)$ these two inequalities have equal slack ($c^0_k-\chi^R_k=r^0_k-\chi^C_k$), so each is equivalent to $(2)$ at index $k$. Hence
$$
\boxed{\ \Pi\text{ satisfies }(2)\ \Longleftrightarrow\
\chi^R_k\le c^0_k\ \text{ and }\ \chi^C_k\le r^0_k\ \text{ for all }k.\ }
\tag{$\star$}
$$
This is the unsymmetric analogue of the symmetric condition $\chi_k\le\zeta_k$: the identity $(3.3)$ plays the role of $\operatorname{nullity}(P_k)=\zeta_k+\chi_k$ and forces the row-view and column-view bounds to coincide. We keep both forms because the explicit construction routes above-arcs through zero columns (needs $\chi^R_k\le c^0_k$) and below-arcs through zero rows (needs $\chi^C_k\le r^0_k$) **independently**, so the $U$-side and $L$-side threads never interact.

**The decisive reformulation: a prefix matching.** Build $L,U$ as a sum of rank-one terms
$$
\Pi=\sum_{t=1}^n \ell_t\,u_t^{\,T},
$$
where $\ell_t$ is column $t$ of $L$ and $u_t^{\,T}$ is row $t$ of $U$. For $L$ lower triangular we need each $\ell_t$ supported on rows $\ge t$; for $U$ upper triangular we need each $u_t$ supported on columns $\ge t$. Because $\Pi$ is $0/1$ with disjoint supports, no telescoping cancellation is needed — it suffices to realize **each matched entry by its own term**. Suppose we choose an **injection**
$$
f:\mathcal M\hookrightarrow\{1,\dots,n\},
\qquad
f(i,j)\le\min(i,j)\ \text{ for every }(i,j)\in\mathcal M,
\tag{3.4}
$$
and set, for each pair $(i,j)$ with $t:=f(i,j)$,
$$
\boxed{\ \ell_t:=e_i,\qquad u_t:=e_j,\ \ }
$$
with $\ell_t=u_t=0$ for indices $t\notin\operatorname{im}f$. Then
$$
LU=\sum_{t=1}^n \ell_t u_t^{\,T}=\sum_{(i,j)\in\mathcal M} e_i e_j^{\,T}=\Pi,
\tag{3.5}
$$
since the matched pairs are exactly the $1$-positions and each appears once. Concretely $(LU)_{i'j'}=\sum_t[\ell_t=e_{i'}][u_t=e_{j'}]$ is nonzero iff some pivot $t$ carries the pair $(i',j')\in\mathcal M$, so $(LU)_{i'j'}=\Pi_{i'j'}$ with no spurious entries. Triangularity is automatic from $(3.4)$: $e_i$ sits at row $i\ge t$ and $e_j$ at column $j\ge t$. The diagonals may carry zeros: $L_{tt}=1$ only if the pair at $t$ has row $t$, and $U_{tt}=1$ only if it has column $t$. Thus

> **the existence of an injection $f$ satisfying $(3.4)$ is exactly the existence of an explicit $LU=\Pi$ with the required triangular supports, over the prime field and hence over every $F$.**

**Existence and explicit form of the injection.** Constraint $(3.4)$ makes each pair $(i,j)$ admissible only on the **prefix** $\{1,\dots,\min(i,j)\}$ of slots. By Hall's theorem in prefix form, a saturating injection exists iff for every threshold $k$,
$$
N_k:=\#\{(i,j)\in\mathcal M:\ \min(i,j)\le k\}\ \le\ k.
\tag{3.6}
$$
A pair has $\min(i,j)\le k$ iff it lies among the $\iota_k$ inside-block pairs, the $\chi^R_k$ crossing above-arcs, or the $\chi^C_k$ crossing below-arcs, so $N_k=\iota_k+\chi^R_k+\chi^C_k$. Using $\iota_k=k-(r^0_k+\chi^R_k)$ from $(3.1)$,
$$
N_k\le k
\iff \chi^R_k+\chi^C_k\le r^0_k+\chi^R_k
\iff \chi^C_k\le r^0_k,
$$
which is $(\star)$. **Hence $(2)$ for $\Pi$ is equivalent to the prefix-Hall condition $(3.6)$, and the injection $f$ exists.**

A concrete deterministic $f$ is the **largest-free-slot greedy**:

> Order $\mathcal M$ by nondecreasing $m=\min(i,j)$ (ties broken arbitrarily). Keeping a set of free slots, initially $\{1,\dots,n\}$, process the pairs in this order and assign to $(i,j)$ the **largest** free slot $t\le m$, removing $t$ from the free set.

This never gets stuck under $(3.6)$. Suppose it failed at a pair with bound $m$: then slots $1,\dots,m$ are all occupied. Every slot $s\le m$ was assigned to a pair of bound $\le m$ (the greedy uses only slots $\le$ the current bound, and bounds increase). Together with the current pair that is $\ge m+1$ pairs with $\min\le m$, i.e. $N_m\ge m+1$, contradicting $(3.6)$. So every pair is assigned, producing $f$ and, via $\ell_{f(i,j)}=e_i$, $u_{f(i,j)}=e_j$, the matrices $L$ (lower triangular) and $U$ (upper triangular) with $LU=\Pi$. $\blacksquare$

**Threads and the role of zero columns/rows.** The greedy makes the **thread** picture transparent and matches the structure of the symmetric proof; the $0/1$ weights collapse its telescoping rank-one identity to a single outer product $e_i e_j^{\,T}$ per matched entry, so no backward $\delta_q,v_q,s_q$ recursion is needed. Reading off the assignment:

- a **diagonal pair** $(t,t)$ has $m=t$ and takes slot $t$: $\ell_t=u_t=e_t$, term $e_t e_t^{\,T}$ ($L_{tt}=U_{tt}=1$), exactly as for a nonzero singleton in the symmetric proof;
- an **above-arc** $(i,j)$, $i<j$, has $m=i$; if index $i$ is **not** also a below-root, slot $i$ is uncontested and the arc is placed at $t=i$: $\ell_i=e_i$, $u_i=e_j$, so $U_{ii}=0$, $U_{ij}=1$;
- a **below-arc** $(i,j)$, $i>j$, has $m=j$; if column $j$ is **not** also an above-root, the arc is placed at $t=j$: $\ell_j=e_i$, $u_j=e_j$, so $L_{jj}=0$, $L_{ij}=1$;
- a **crossing index** $t$ — simultaneously the root of an above-arc $(t,j)$, $j>t$, and of a below-arc $(i,t)$, $i>t$ — can serve only one of the two through slot $t$. The greedy keeps slot $t$ for one and **reroutes** the other onto an earlier free slot, supplied by a **zero column** (for a displaced above-arc) or a **zero row** (for a displaced below-arc). The arcs reusing one freed slot form a thread rooted at that zero column or zero row.

The feasibility of the reroutings is exactly $(\star)$: at the cut $k$ the above-arcs forced off their own roots must occupy distinct earlier slots, of which only the $c^0_k$ zero-column slots $\le k$ are available, whence $\chi^R_k\le c^0_k$; dually $\chi^C_k\le r^0_k$ governs the below-side. Because the above-side threads live in column slots and the below-side threads in row slots, **the $U$-side and $L$-side threads are independent and never interact**.

**Conclusion of Step 3.** For a partial permutation $\Pi$ satisfying $(2)$, the greedy of $(3.4)$–$(3.6)$ produces explicit lower-triangular $L$ and upper-triangular $U$ (with $0/1$ entries, hence over the prime field and every $F$) with $LU=\Pi$. Combined with Step 1 ($A=M\Pi N$) and Step 2 (the three nullities are invariant under invertible lower/upper-triangular pre/post-multiplication, so $A$ satisfies $(2)$ iff $\Pi$ does),
$$
A=M\,\Pi\,N=M\,(LU)\,N=(ML)\,(UN),
$$
where $ML$ is lower triangular and $UN$ is upper triangular. Hence $A$ has an unpivoted $LU$ factorization with zeros allowed on both diagonals. This proves sufficiency over any commutative field $F$. $\blacksquare$

### Worked crossing example

Let $n=4$ and let $\Pi$ have the two matched pairs $(2,4)$ (an above-arc) and $(4,2)$ (a below-arc); rows $1,3$ and columns $1,3$ are zero:
$$
\Pi=
\begin{bmatrix}
0&0&0&0\\
0&0&0&1\\
0&0&0&0\\
0&1&0&0
\end{bmatrix}.
$$
The counts are
$$
\begin{array}{c|cccc}
k & 1 & 2 & 3 & 4\\\hline
\chi^R_k & 0 & 1 & 1 & 0\\
c^0_k & 1 & 1 & 2 & 2\\
\chi^C_k & 0 & 1 & 1 & 0\\
r^0_k & 1 & 1 & 2 & 2
\end{array}
$$
so $(\star)$ holds ($\chi^R_k\le c^0_k$ and $\chi^C_k\le r^0_k$ at every $k$). Index $2$ is a **crossing index**: both arcs have root $2$, so a single slot cannot carry both. The greedy keeps slot $2$ for the below-arc and reroutes the above-arc onto the spare slot $1$ — a zero row **and** a zero column. This gives
$$
L=
\begin{bmatrix}
0&0&0&0\\
1&0&0&0\\
0&0&0&0\\
0&1&0&0
\end{bmatrix},
\qquad
U=
\begin{bmatrix}
0&0&0&1\\
0&1&0&0\\
0&0&0&0\\
0&0&0&0
\end{bmatrix},
\qquad
LU=
\begin{bmatrix}
0&0&0&0\\
0&0&0&1\\
0&0&0&0\\
0&1&0&0
\end{bmatrix}=\Pi .
$$
Here slot $2$ carries the below-arc ($\ell_2=e_4,\ u_2=e_2$, giving $L_{42}=U_{22}=1$) and slot $1$ carries the above-arc rerouted onto the zero row/column ($\ell_1=e_2,\ u_1=e_4$, giving $L_{21}=U_{14}=1$). As in the symmetric case, zero diagonal entries in **both** $L$ and $U$ are essential: the leading zero index $1$ roots the thread that realizes the crossing arc.

### End-to-end numerical check

Script: `/Users/darve/git_repositories/existence_of_lu/referee_ignore/verify_unsym_endtoend.py` (run with `/Users/darve/miniconda3/bin/python3`). Ranks are computed by **exact** fraction-free Gaussian elimination over $\mathbb Q$ (no floating-point tolerance), so the checks cannot be fooled by rank artifacts. For each test it computes the three nullities of $(2)$ directly, checks them against $(3.1)$–$(3.2)$ and the bookkeeping identity $(3.3)$, checks $(2)\Leftrightarrow(\star)$, and — for every $\Pi$ satisfying $(2)$ — runs the largest-free-slot greedy to build $L,U$ and asserts $L$ lower-triangular, $U$ upper-triangular, and $L\cdot U=\Pi$ exactly over the integers.

```
[A] nullity formulas (random 4000):                       failures = 0
[B] equivalence (2)<=>combinatorial (random 4000):        failures = 0
[B2] exhaustive n=1..4 equivalence (252 partial perms):   failures = 0
[B2] exhaustive n=1..4 formulas (252 partial perms):      failures = 0
[C] end-to-end L@U==Pi (random satisfying, 3778 built):   failures = 0
[C2] exhaustive n=1..5 end-to-end (948 satisfying,
                                   152 with a crossing):  failures = 0
hand cases:
  swap [[0,1],[1,0]]:   cond(2) = False, combinatorial = False   (expect False)
  antidiagonal n=3:     cond(2) = False                          (expect False)
  zero 3x3 / identity:  cond(2) = True
  above-arc (1,3):      cond(2) = True,  L@U == Pi                (zero column rooted)
  below-arc (3,1):      cond(2) = True,  L@U == Pi                (zero row rooted)
  crossing n=4:         cond(2) = True,  L@U == Pi                (rerouted via slot 1)
  two-above-arc thread: cond(2) = True,  L@U == Pi
TOTAL FAILURES: 0   (ALL ASSERTIONS PASSED)
```

In particular the forbidden swap $\bigl[\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\bigr]$ correctly **fails** $(2)$ — here $\chi^R_1=1>c^0_1=0$ — and so it has no zero-diagonal-allowed $LU$ factorization, while every partial permutation satisfying $(2)$, including all $152$ instances with a genuine crossing index among the $948$ satisfying cases for $n\le5$, factors exactly as $L\cdot U=\Pi$ with $L$ lower triangular and $U$ upper triangular. This confirms the explicit construction end to end.
