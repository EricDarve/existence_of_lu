For a square matrix $A\in F^{n\times n}$, the exact criterion is not just “all leading principal minors are nonzero.” That classical condition is the right answer only in the nonsingular case. For arbitrary square matrices, including singular ones, the exact necessary-and-sufficient condition is rank-based. Okunev and Johnson proved that $A$ has an $LU$ factorization with no row or column permutations, $A=LU$, iff for every $k=1,\dots,n$,
$$
\operatorname{rank}\big(A_{1:k,,1:k}\big)+k
\ge
\operatorname{rank}\big(A_{1:k,,:}\big)
+
\operatorname{rank}\big(A_{:,,1:k}\big).
$$
Here $A_{1:k,1:k}$ is the leading $k\times k$ principal block, $A_{1:k,:}$ the first $k$ rows, and $A_{:,1:k}$ the first $k$ columns. The $k=n$ case is automatic, so one usually checks $k=1,\dots,n-1$. The same condition can be written in nullity form as
$$
\operatorname{null}\big(A_{1:k,1:k}\big)
\le
\operatorname{null}\big(A_{:,1:k}\big)+\operatorname{null}\big(A_{1:k,:}^{,T}\big).
$$
This is Theorem 1 in the Okunev–Johnson work; a later report by Darve states the equivalent nullity form and notes that it is equivalent to Okunev–Johnson. ([arXiv][1])

A quick sanity check: the matrix
$$
\begin{bmatrix}0&1\\0&0\end{bmatrix}
$$
already is upper triangular, so it certainly has an $LU$ factorization ($L=I,\ U=A$), even though its first leading principal minor is $0$. So the determinant test is not the full story once singular matrices are allowed.

## Proof of the criterion

Write
$$
P_k:=A_{1:k,1:k},\qquad R_k:=A_{1:k,:},\qquad C_k:=A_{:,1:k}.
$$

### Necessity

Assume $A=LU$ with $L$ lower triangular and $U$ upper triangular. Partition conformally at size $k$:
$$
A=\begin{bmatrix}P_k & *\\ * & *\end{bmatrix},\quad
L=\begin{bmatrix}L_k & 0\\ * & *\end{bmatrix},\quad
U=\begin{bmatrix}U_k & *\\ 0 & *\end{bmatrix}.
$$
Then
$$
P_k=L_kU_k,\qquad
R_k=L_k\,[\,U_k\ \ *\,],\qquad
C_k=\begin{bmatrix}L_k\\ *\end{bmatrix}U_k.
$$
So
$$
\operatorname{rank}(R_k)\le \operatorname{rank}(L_k),\qquad
\operatorname{rank}(C_k)\le \operatorname{rank}(U_k).
$$
Also, by Sylvester’s rank inequality applied to $P_k=L_kU_k$,
$$
\operatorname{rank}(P_k)\ge \operatorname{rank}(L_k)+\operatorname{rank}(U_k)-k.
$$
Combining the three inequalities gives
$$
\operatorname{rank}(P_k)+k\ge \operatorname{rank}(R_k)+\operatorname{rank}(C_k),
$$
which is the required condition.

### Sufficiency

We prove the converse by induction on $n$.

For $n=1$, it is trivial.

Now assume the statement is true for $(n-1)\times(n-1)$ matrices, and write
$$
A=\begin{bmatrix}
a & r^T\
c & A_{22}
\end{bmatrix},
$$
where $a=a_{11}$, $r^T=A_{1,2:n}$, and $c=A_{2:n,1}$.

When $a=0$, the condition with $k=1$ says
$$
0+1\ge \operatorname{rank}(R_1)+\operatorname{rank}(C_1).
$$
Since each rank is $0$ or $1$, at least one of the first row and first column must be zero.

### Case 1: $a\neq 0$

Set
$$
\ell:=c,\qquad u^T:=a^{-1}r^T,
$$
and define
$$
B:=A_{22}-\ell u^T.
$$
Now form
$$
\widetilde A
:=
A-\begin{bmatrix}a \\ \ell\end{bmatrix}
\begin{bmatrix}0&u^T\end{bmatrix}
=
\begin{bmatrix}
a&0 \\
\ell & B
\end{bmatrix}.
$$
This is obtained from $A$ by elementary column operations using column 1, so the ranks of all relevant leading blocks, first-row blocks, and first-column blocks are unchanged; hence $\widetilde A$ satisfies the same rank condition as $A$.

For each $k\le n-1$, because the $(1,1)$-entry of $\widetilde A$ is nonzero and the rest of the first row is zero,
$$
\operatorname{rank}(P_{k+1}(\widetilde A))=1+\operatorname{rank}(P_k(B)),
$$
and similarly
$$
\operatorname{rank}(R_{k+1}(\widetilde A))=1+\operatorname{rank}(R_k(B)),\qquad
\operatorname{rank}(C_{k+1}(\widetilde A))=1+\operatorname{rank}(C_k(B)).
$$
Applying the condition to $\widetilde A$ at level $k+1$ yields the condition for $B$ at level $k$. So $B$ satisfies the criterion, and by the induction hypothesis $B=L'U'$ with $L'$ lower triangular and $U'$ upper triangular.

Therefore
$$
A=
\begin{bmatrix}
a&0 \\
\ell&L'
\end{bmatrix}
\begin{bmatrix}
1&u^T \\
0&U'
\end{bmatrix},
$$
which is an $LU$ factorization.

### Case 2: $a=0$, first row is zero, first column is not zero

Let $i$ be the smallest index with $c_i\neq 0$. Set
$$
\ell:=c,\qquad
u^T:=c_i^{-1}A_{i,,2:n},
\qquad
B:=A_{22}-\ell u^T.
$$
Then
$$
\widetilde A
:=
A-\begin{bmatrix}0 \\ \ell\end{bmatrix}
\begin{bmatrix}0&u^T\end{bmatrix}
=
\begin{bmatrix}
0&0 \\
\ell&B
\end{bmatrix}.
$$
Again $\widetilde A$ is obtained from $A$ by elementary column operations, so it satisfies the same rank condition. By construction, row $i$ of $\widetilde A$ is
$$
[c_i,0,\dots,0].
$$

For any $k\le n-1$, column 1 of $C_{k+1}(\widetilde A)$ is independent of columns $2,\dots,k+1$, hence
$$
\operatorname{rank}(C_{k+1}(\widetilde A))=1+\operatorname{rank}(C_k(B)).
$$

If $i\le k+1$, then the special row $([c_i,0,\dots,0])$ is visible inside the first $(k+1)$ rows, so
$$
\operatorname{rank}(P_{k+1}(\widetilde A))=1+\operatorname{rank}(P_k(B)),
\qquad
\operatorname{rank}(R_{k+1}(\widetilde A))=1+\operatorname{rank}(R_k(B)).
$$

If $i>k+1$, then inside the first $(k+1)$ rows the first column is still zero, and the first row is zero too, so deleting that first row and column does not change rank:
$$
\operatorname{rank}(P_{k+1}(\widetilde A))=\operatorname{rank}(P_k(B)),
\qquad
\operatorname{rank}(R_{k+1}(\widetilde A))=\operatorname{rank}(R_k(B)).
$$

In either subcase, the rank condition for $\widetilde A$ at size $k+1$ becomes exactly the rank condition for $B$ at size $k$. Hence $B$ satisfies the criterion. By induction, $B=L'U'$, and then
$$
A=
\begin{bmatrix}
0&0 \\
\ell&L'
\end{bmatrix}
\begin{bmatrix}
1&u^T \\
0&U'
\end{bmatrix}.
$$

### Case 3: $a=0$, first column is zero, first row is not zero

The condition is symmetric under transpose:
$$
\operatorname{rank}(P_k(A^T))=\operatorname{rank}(P_k(A)),\quad
\operatorname{rank}(R_k(A^T))=\operatorname{rank}(C_k(A)),\quad
\operatorname{rank}(C_k(A^T))=\operatorname{rank}(R_k(A)).
$$
So $A^T$ satisfies the same criterion and falls under Case 2. Hence $A^T=LU$ for some lower $L$ and upper $U$. Transposing gives
$$
A=U^TL^T,
$$
and $U^T$ is lower triangular while $L^T$ is upper triangular.

### Case 4: the first row and first column are both zero

If $A=0$, we are done.

Otherwise, let $j>1$ be the smallest index such that column $j$ is nonzero, and form $\widehat A$ by replacing the first column of $A$ by column $j$ of $A$. Then $\widehat A$ has first row zero and first column nonzero.

We claim $\widehat A$ satisfies the rank condition. For $k<j$, columns $2,\dots,k$ of $\widehat A$ are zero and row 1 is zero, so
$$
\operatorname{rank}(R_k(\widehat A))\le k-1,\qquad
\operatorname{rank}(C_k(\widehat A))=1.
$$
Hence
$$
\operatorname{rank}(P_k(\widehat A))+k \ge k \ge (k-1)+1
$$
and the condition holds automatically. For $k\ge j$, column 1 of $\widehat A$ is just a duplicate of an already existing column among the first $k$ columns, while column 1 of $A$ was zero, so the ranks of $P_k$, $R_k$, and $C_k$ are unchanged; thus the condition transfers from $A$ to $\widehat A$.

Now apply Case 2 to $\widehat A$. It produces a column $\ell$ and a row $u^T$ such that
$$
B:=\widehat A_{22}-\ell u^T
$$
satisfies the criterion. But $\widehat A_{22}=A_{22}$, and since the first row and column of $A$ are zero, $A_{12}=0$ and $A_{21}=0$. By induction, $B=L'U'$, so
$$
A=
\begin{bmatrix}
0&0 \\
\ell&L'
\end{bmatrix}
\begin{bmatrix}
0&u^T \\
0&U'
\end{bmatrix}.
$$

That completes the induction, so the rank condition is sufficient.

## Nonsingular case: the familiar corollary

If $A$ is invertible, then any subset of its columns is linearly independent and any subset of its rows is linearly independent. Therefore
$$
\operatorname{rank}(R_k)=k,\qquad \operatorname{rank}(C_k)=k.
$$
So the general criterion becomes
$$
\operatorname{rank}(P_k)=k\qquad (k=1,\dots,n),
$$
which is equivalent to
$$
\det(P_k)\neq 0\qquad (k=1,\dots,n).
$$
Thus for an invertible matrix, $A$ has an $LU$ factorization without pivoting iff all leading principal minors are nonzero. That is the classical textbook criterion; the general rank condition is its singular/rank-deficient extension. ([arXiv][2])

## Paper where the proof appears

The original freely accessible source is:

Pavel Okunev and Charles R. Johnson, *Necessary And Sufficient Conditions For Existence of the LU Factorization of an Arbitrary Matrix* (arXiv:math/0506382, submitted 2005). The theorem is stated there as Theorem 1, and the proof is constructive. ([arXiv][3])

The archival journal publication is:

Charles R. Johnson and Pavel Okunev, *Characterization of the existence of an (L)-(U) factorization*, *Advances in Operator Theory* 10, article 50 (2025). Its abstract states the same rank characterization and says the proof gives an explicit construction. ([Springer][4])

A newer source with a cleaner nullity-based restatement and “simpler, constructive proofs” is:

Eric Darve, *Necessary and Sufficient Conditions for the Existence of an LU Factorization for General Rank Deficient Matrices* (technical report, 2026). It explicitly says its condition is equivalent to Okunev–Johnson. ([arXiv][2])

[1]: https://arxiv.org/pdf/math/0506382.pdf "https://arxiv.org/pdf/math/0506382.pdf"
[2]: https://arxiv.org/pdf/2601.07791 "https://arxiv.org/pdf/2601.07791"
[3]: https://arxiv.org/abs/math/0506382 "https://arxiv.org/abs/math/0506382"
[4]: https://link.springer.com/article/10.1007/s43036-024-00400-2 "https://link.springer.com/article/10.1007/s43036-024-00400-2"
