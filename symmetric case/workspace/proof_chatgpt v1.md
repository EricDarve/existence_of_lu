For a real symmetric matrix $A$, there is a direct proof that avoids invoking the Van Dooren–Strang theorem as a black box. The proof below is self-contained: it uses an explicit bordered lower-triangular congruence to a sparse symmetric skeleton, and then factors that skeleton directly by a greedy threading argument.

Let
$$
A_k := A(1:k,1:k),
\qquad
C_k := A(:,1:k).
$$

## The symmetric criterion

For $A=A^T\in \mathbb R^{n\times n}$, the following are equivalent:

1. There exist a lower triangular matrix $L$ and a diagonal matrix $D$ such that
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

## Proof

### Necessity

Assume
$$
A=LDL^T,
$$
and partition
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
Applying Sylvester’s inequality to
$$
A_k = L_{11}(D_1L_{11}^T)
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
So
$$
\operatorname{rank}(A_k)+k \ge 2\operatorname{rank}(C_k),
$$
equivalently
$$
\operatorname{nullity}(A_k)\le 2\,\operatorname{nullity}(C_k).
$$

### Sufficiency

I will prove sufficiency in three short steps.

#### Step 1: reduce $A$ by elementary symmetric elimination to a signed matching matrix

Call a symmetric matrix $P$ a **signed matching matrix** if:

- each row and each column has at most one nonzero entry,
- each diagonal entry belongs to $\{0,\pm1\}$,
- each off-diagonal nonzero entry equals $1$.

So $P$ is a disjoint union of isolated zeros, diagonal $+1$ or $-1$, and exchange pairs
$$
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
$$
placed at arbitrary indices.

I claim that one can construct, inductively on $k$, an invertible lower triangular matrix $M_k$ and a signed matching matrix $P_k$ such that
$$
A_k = M_k P_k M_k^T.
$$

For $k=1$ this is obvious.

Now assume
$$
A_{k-1}=M_{k-1}P_{k-1}M_{k-1}^T,
$$
where $M_{k-1}$ is invertible lower triangular and $P_{k-1}$ is a signed matching matrix. Write
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

Because $P_{k-1}$ is a signed matching matrix, $P_{k-1}^2$ is the diagonal projector onto $\operatorname{range}(P_{k-1})$. Define
$$
\xi := P_{k-1}c,
\qquad
e := (I-P_{k-1}^2)c,
\qquad
f := b-c^TP_{k-1}c.
$$
Then $P_{k-1}e=0$, and
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

So it remains to replace
$$
\begin{bmatrix}
P_{k-1} & e\\
e^T & f
\end{bmatrix}
$$
by another signed matching matrix via a lower triangular congruence.

If $e=0$, then the block is simply $P_{k-1}\oplus [f]$. Over $\mathbb R$, by scaling the last coordinate, this is congruent to
$$
P_{k-1}\oplus [\operatorname{sgn}(f)],
$$
with $\operatorname{sgn}(0)=0$. So in this case we just append a diagonal entry $0,\pm1$.

If $e\neq 0$, let $i$ be the first nonzero component of $e$. Since $P_{k-1}e=0$, index $i$ must be isolated in $P_{k-1}$: if row/column $i$ carried any nonzero of $P_{k-1}$, then $(P_{k-1}e)$ would have a nonzero entry forced by $e_i\neq 0$. Now define
$$
L_{\mathrm{up}}
=
\begin{bmatrix}
I + (e-u_i)u_i^T & 0\\[2mm]
\frac f2\,u_i^T & 1
\end{bmatrix},
$$
where $u_i$ is the $i$th standard basis vector. Because $i$ is the first nonzero of $e$, this matrix is lower triangular. A direct multiplication gives
$$
\begin{bmatrix}
P_{k-1} & e\\
e^T & f
\end{bmatrix}
=
L_{\mathrm{up}}
\begin{bmatrix}
P_{k-1} & u_i\\
u_i^T & 0
\end{bmatrix}
L_{\mathrm{up}}^T.
$$
The middle matrix is again a signed matching matrix: it simply connects the new index $k$ to the previously isolated index $i$.

Thus, by induction,
$$
A = MPM^T
$$
for some invertible lower triangular $M$ and some signed matching matrix $P$.

#### Step 2: the nullity condition transfers from $A$ to $P$

If
$$
B = TAT^T
$$
with $T$ invertible lower triangular, then for every $k$,
$$
B_k = T_k A_k T_k^T,
\qquad
B(:,1:k) = T\,A(:,1:k)\,T_k^T,
$$
where $T_k$ is the leading $k\times k$ block of $T$. Hence
$$
\operatorname{nullity}(B_k)=\operatorname{nullity}(A_k),
\qquad
\operatorname{nullity}(B(:,1:k))=\operatorname{nullity}(A(:,1:k)).
$$
So $A$ satisfies
$$
\operatorname{nullity}(A_k)\le 2\,\operatorname{nullity}(C_k)
$$
if and only if $P$ satisfies
$$
\operatorname{nullity}(P_k)\le 2\,\operatorname{nullity}(P(:,1:k)).
$$

#### Step 3: for $P$, the condition is combinatorial and gives an explicit $LDL^T$

For a signed matching matrix $P$, define:

- $\zeta_k$ = the number of isolated zero indices among $\{1,\dots,k\}$,
- $\chi_k$ = the number of exchange pairs $(i,j)$ with $i\le k<j$.

Then:

- in $P_k$, each isolated zero contributes one null direction, and each crossing pair contributes one more null direction, so
  $$
  \operatorname{nullity}(P_k)=\zeta_k+\chi_k;
  $$
- in $P(:,1:k)$, the only zero columns are the isolated-zero columns, so
  $$
  \operatorname{nullity}(P(:,1:k))=\zeta_k.
  $$

Therefore the hypothesis becomes exactly
$$
\chi_k \le \zeta_k
\qquad (k=1,\dots,n). \tag{$*$}
$$

Now scan the indices from left to right.

Each isolated zero creates one free **thread**.  
When an exchange pair $(i,j)$ starts at $i$, assign it to any free thread.  
When it ends at $j$, that thread becomes free again.

At step $k$, the busy threads are exactly the crossing pairs counted by $\chi_k$, while the total threads created so far are the isolated zeros counted by $\zeta_k$. Thus condition $(*)$ guarantees that a free thread always exists. Hence the exchange pairs split into disjoint threads of the form
$$
r<i_1<j_1<i_2<j_2<\cdots<i_m<j_m,
$$
where $r$ is the isolated zero that roots the thread.

Now fix one thread
$$
r<i_1<j_1<\cdots<i_m<j_m.
$$
Define
$$
s_q := e_{i_q}+e_{i_{q+1}}+\cdots+e_{i_m},
\qquad
s_{m+1}:=0,
$$
and define the thread-columns
$$
\ell_r := s_1,
\qquad
\ell_{i_q}:=s_q+e_{j_q},
\qquad
\ell_{j_q}:=s_{q+1}+e_{j_q}.
$$
Because $j_q<i_{q+1}$, every $\ell_\alpha$ is supported only in rows $\ge \alpha$, so these really are columns of a lower triangular matrix.

Now choose diagonal signs
$$
d_r=-1,
\qquad
d_{i_q}=+1,
\qquad
d_{j_q}=-1.
$$
Then
$$
\ell_{i_q}\ell_{i_q}^T-\ell_{j_q}\ell_{j_q}^T
=
s_qs_q^T-s_{q+1}s_{q+1}^T
+
e_{i_q}e_{j_q}^T+e_{j_q}e_{i_q}^T.
$$
Summing over $q$ and subtracting $\ell_r\ell_r^T=s_1s_1^T$, the $s_qs_q^T$ terms telescope, leaving
$$
-\ell_r\ell_r^T
+
\sum_{q=1}^m
\bigl(\ell_{i_q}\ell_{i_q}^T-\ell_{j_q}\ell_{j_q}^T\bigr)
=
\sum_{q=1}^m
\bigl(e_{i_q}e_{j_q}^T+e_{j_q}e_{i_q}^T\bigr).
$$
So one thread produces exactly the exchange pairs on that thread, and nothing else.

Diagonal $+1$ and $-1$ entries of $P$ are trivial one-column factors $\pm e_te_t^T$. Unused isolated zeros contribute zero columns. Since different threads are disjoint in indices, the contributions of distinct threads do not interact. Therefore
$$
P=SES^T
$$
for some lower triangular $S$ and diagonal $E$.

Finally,
$$
A=MPM^T=(MS)E(MS)^T,
$$
and $MS$ is lower triangular. Hence $A$ has an $LDL^T$ factorization.

This proves sufficiency.

## Check on
$$
A=
\begin{bmatrix}
0&0&0\\
0&0&1\\
0&1&0
\end{bmatrix}.
$$

Here $A$ itself is already a signed matching matrix: index $1$ is an isolated zero, and $(2,3)$ is one exchange pair. So
$$
\zeta_1=\zeta_2=\zeta_3=1,
\qquad
\chi_1=0,\ \chi_2=1,\ \chi_3=0.
$$
Thus $\chi_k\le \zeta_k$ for every $k$, exactly as required.

There is one thread:
$$
1<2<3.
$$
The thread construction gives
$$
L=
\begin{bmatrix}
0&0&0\\
1&1&0\\
0&1&1
\end{bmatrix},
\qquad
D=\operatorname{diag}(-1,1,-1),
$$
and indeed
$$
LDL^T=
\begin{bmatrix}
0&0&0\\
0&0&1\\
0&1&0
\end{bmatrix}
=A.
$$

So the proof is valid on your test matrix. The example also shows why allowing $l_{11}=0$ is essential: the leading zero index is the root of the thread that realizes the later exchange pair $(2,3)$.

## References

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