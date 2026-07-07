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



# Proof from Gemini

To determine the necessary and sufficient conditions for a symmetric matrix $A \in F^{n \times n}$ to have a factorization $A = LDL^T$ (where $L$ is lower triangular and $D$ is diagonal), we analyze the symmetric Gaussian elimination process and the degrees of freedom provided by the lower triangular structure of $L$.

### Necessary and Sufficient Condition
A symmetric matrix $A$ has an $L D L^T$ factorization if and only if:
**Let $k \in \{1, \dots, n\}$ be the smallest integer such that the leading principal minor $\det(A[1:k, 1:k]) = 0$. (If no such $k$ exists, the condition is vacuously satisfied). Then the $k$-th row of $A$ must be a linear combination of the first $k-1$ rows of $A$.**

Equivalently, this means that if the standard symmetric Gaussian elimination without pivoting encounters a zero pivot at step $k$ for the first time, the *entire* first row and column of the remaining Schur complement must be exactly zero.

---

### Proof of Necessity
Suppose $A = L D L^T$ for some lower triangular $L$ and diagonal $D$. Let $A_m = A[1:m, 1:m]$. 
Because $L$ is lower triangular, the upper-left $m \times m$ block of $A$ is completely determined by the upper-left $m \times m$ blocks of $L$ and $D$:
$$A_m = L_m D_m L_m^T$$
Let $k$ be the smallest index such that $\det(A_k) = 0$. Since $\det(A_{k-1}) \neq 0$, $L_{k-1}$ and $D_{k-1}$ must be non-singular, meaning $L_{ii} \neq 0$ and $d_i \neq 0$ for all $i < k$. 

For $A_k$ to be singular while $A_{k-1}$ is non-singular, the $k$-th diagonal entry of $L_k D_k L_k^T$ must drop rank. Because $L_k$ is lower triangular, its determinant is the product of its diagonal entries. Thus, we must have $L_{kk}^2 d_k = 0$, which implies $L_{kk} d_k = 0$.

Now consider the $k$-th row of $A$ from the $k$-th column onward. For any $j \ge k$, the entry $A_{k, j}$ is given by the inner product of the $k$-th and $j$-th rows of $L D^{1/2}$ (conceptually). Specifically:
$$A_{k, j} = \sum_{m=1}^k L_{km} d_m L_{jm}$$
Because $L_{kk} d_k = 0$, the $m=k$ term vanishes. Thus, the sum only goes up to $k-1$:
$$A_{k, j} = \sum_{m=1}^{k-1} L_{km} d_m L_{jm}$$
This is exactly the formula for the elements in the span of the first $k-1$ rows of $A$. Therefore, the $k$-th row of $A$ is strictly a linear combination of the first $k-1$ rows of $A$. 

---

### Proof of Sufficiency
If $\det(A_k) \neq 0$ for all $k$, standard symmetric Gaussian elimination succeeds without pivoting, yielding $A = L D L^T$ with a unit lower triangular $L$. 

Now suppose $k$ is the first index where $\det(A_k) = 0$, and the $k$-th row of $A$ is a linear combination of the first $k-1$ rows. 
We can perform standard elimination for the first $k-1$ steps. After $k-1$ steps, the Schur complement $S^{(k)}$ (of size $(n-k+1) \times (n-k+1)$) represents the remaining matrix to be factored. Because the $k$-th row of $A$ was in the span of the first $k-1$ rows, the elimination of those $k-1$ rows completely zeroes out the $k$-th row (and by symmetry, the $k$-th column) of the Schur complement. 

Thus, $S^{(k)}$ has the block form:
$$S^{(k)} = \begin{pmatrix} 0 & \mathbf{0} \\ \mathbf{0} & B \end{pmatrix}$$
where $B$ is an $(n-k) \times (n-k)$ symmetric matrix. 

We must factor $S^{(k)}$ using the remaining $n-k+1$ columns of $L$. Because the first row/col of $S^{(k)}$ is zero, we can set $L_{kk} = 0$ and choose $d_k \neq 0$ (e.g., $d_k = 1$). This choice perfectly satisfies the $0$ boundaries of $S^{(k)}$, but crucially leaves the rest of the $k$-th column of $L$ — let's call it $\mathbf{v} \in F^{n-k}$ — completely unconstrained.

This provides us with a "free" column. We are now tasked with factoring the $(n-k) \times (n-k)$ matrix $B$ using $n-k+1$ columns (the free column $\mathbf{v}$ plus the remaining $n-k$ columns of $L$). 
We want to express $B$ as:
$$B = d_k \mathbf{v} \mathbf{v}^T + L_B D_B L_B^T$$
Because we have one more column than the dimension of $B$, we have an underdetermined system. At every subsequent row $i$ of $B$, we will have $i+1$ variables to satisfy $i$ equations. This extra degree of freedom allows us to construct an isotropic vector at any point we encounter another zero pivot, bypassing the rigid constraints of standard $LU$ or Cholesky factorizations. 

By pairing the free column with the current column, we can locally form symmetric outer products $\frac{1}{2}(\mathbf{u}\mathbf{w}^T + \mathbf{w}\mathbf{u}^T)$ which can span any symmetric matrix. The "free column" property regenerates itself after bypassing a zero diagonal, ensuring that the remainder of the matrix $B$ can be factored unconditionally, regardless of its principal minors. Thus, the factorization $A = L D L^T$ is guaranteed to be completed.

