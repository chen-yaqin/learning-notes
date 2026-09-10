---
title: 4. Kernels, Gaussian Processes, and Conformal Prediction
description: STAT615 lectures 17-23 — kernel constructions, RKHS, representer theorem, spectral expansions, and calibrated prediction sets.
aliases:
  - Kernels and RKHS
  - Representer Theorem and Kernel Regression
  - Gaussian Processes
  - Conformal Prediction
tags:
  - statistical-learning
  - stat615
---

[[3 Linear Classifiers and Support Vector Machines|Previous: linear classifiers and SVMs]] · [[Statistical Learning/index|Course overview]]

Source: [Lectures 17-23, pp. 63-90](615.pdf#page=63).

Part 3 showed that an SVM can work entirely through feature-space inner products. This part asks which functions represent such inner products, how they define spaces of predictors, and how the same kernels describe random functions. The course ends by calibrating prediction sets.

## 1. Kernels and their basic constructions

_Lecture 17; [pp. 63-65](615.pdf#page=63)._

A real function $K:\mathcal X\times\mathcal X\to\mathbb R$ is a **positive semidefinite kernel** if, for every finite collection $x_1,\ldots,x_m$, its Gram matrix

$$
G_{ij}=K(x_i,x_j)
$$

is symmetric and positive semidefinite: $c^\top Gc\ge0$ for every $c\in\mathbb R^m$. The requirement concerns every finite collection, not just the current training sample.

### Inner products define kernels

If $K(x,z)=\langle\Phi(x),\Phi(z)\rangle_{\mathcal H}$, symmetry follows from the inner product. Moreover,

$$
\begin{aligned}
c^\top Gc
&=\sum_{i,j}c_ic_j\langle\Phi(x_i),\Phi(x_j)\rangle_{\mathcal H}\\
&=\left\langle\sum_i c_i\Phi(x_i),\sum_j c_j\Phi(x_j)\right\rangle_{\mathcal H}\\
&=\left\|\sum_i c_i\Phi(x_i)\right\|_{\mathcal H}^2\ge0.
\end{aligned}
$$

Thus every feature map supplies a valid kernel.

### Closure properties

The lecture uses several ways to construct new kernels.

**Nonnegative sums.** If $K_1,K_2$ are kernels and $a_1,a_2\ge0$, then $a_1K_1+a_2K_2$ is a kernel because

$$
c^\top(a_1G_1+a_2G_2)c=a_1c^\top G_1c+a_2c^\top G_2c\ge0.
$$

**Scaling by a function.** For any real function $a$, the kernel $\widetilde K(x,z)=a(x)a(z)K(x,z)$ has Gram matrix $DGD$, where $D_{ii}=a(x_i)$. Its quadratic form is $(Dc)^\top G(Dc)\ge0$.

**Products.** $K_1(x,z)K_2(x,z)$ is also a kernel. On any finite sample, factor $G_1,G_2$ as Gram matrices of vectors $u_i,v_i$. Then

$$
(G_1)_{ij}(G_2)_{ij}
=\sum_{a,b}(u_{ia}v_{ib})(u_{ja}v_{jb}),
$$

which is itself a Gram matrix. This is the positive-semidefinite property of the entrywise product.

**Pointwise limits.** A finite pointwise limit of kernels is a kernel: each finite quadratic form is the limit of nonnegative quadratic forms.

### Polynomial, exponential, and Gaussian examples

Start with $K(x,z)=x^\top z$. Products and nonnegative sums show that

$$
(x^\top z)^m,\qquad \sum_{m=0}^M a_m(x^\top z)^m\quad(a_m\ge0)
$$

are kernels. Taking the limit of Taylor polynomials gives

$$
e^{x^\top z}=\sum_{m=0}^{\infty}\frac{(x^\top z)^m}{m!}.
$$

Now expand the squared distance:

$$
\|x-z\|^2=\|x\|^2-2x^\top z+\|z\|^2.
$$

For $\ell>0$,

$$
e^{-\|x-z\|^2/(2\ell^2)}
=e^{-\|x\|^2/(2\ell^2)}
e^{-\|z\|^2/(2\ell^2)}
e^{x^\top z/\ell^2}.
$$

The last factor is an exponential inner-product kernel, and the first two implement scaling by a function. This proves that the Gaussian/RBF kernel is valid.

## 2. Hilbert spaces and continuous evaluation

_Lecture 18; [pp. 66-68](615.pdf#page=66)._

A **Hilbert space** is an inner-product vector space complete in its induced norm: every Cauchy sequence converges to an element of the space. Its inner product gives

$$
\|u\|_{\mathcal H}=\sqrt{\langle u,u\rangle_{\mathcal H}},
\qquad
|\langle u,v\rangle_{\mathcal H}|\le\|u\|_{\mathcal H}\|v\|_{\mathcal H}.
$$

The second inequality is Cauchy-Schwarz.

The lecture examples are:

- $\mathbb R^m$, with $\langle u,v\rangle=\sum_iu_iv_i$.
- $\ell^2$, the sequences with $\sum_i u_i^2<\infty$, using the same infinite-sum inner product.
- $L^2$, square-integrable functions with $\langle f,g\rangle=\int fg$. Its elements are equivalence classes up to equality almost everywhere.

A linear functional $L:\mathcal H\to\mathbb R$ is continuous exactly when $|L(f)|\le C\|f\|_{\mathcal H}$ for some finite $C$. Inner products with a fixed $g$ are continuous by Cauchy-Schwarz.

The **Riesz representation theorem**, stated in the lecture, says the converse: every continuous linear functional has a unique representation $L(f)=\langle f,g\rangle_{\mathcal H}$.

### The reproducing property

An **RKHS** is a Hilbert space of actual functions on $\mathcal X$ for which evaluation $L_x(f)=f(x)$ is continuous for every $x$.

Riesz then gives an element $K_x\in\mathcal H$ satisfying

$$
f(x)=\langle f,K_x\rangle_{\mathcal H}.
$$

Take $f=K_z$ to get $K_z(x)=\langle K_z,K_x\rangle_{\mathcal H}$. Define

$$
K(x,z)=\langle K_x,K_z\rangle_{\mathcal H}.
$$

It is a kernel by the inner-product calculation above, and $K_x=K(\cdot,x)$. Therefore

$$
f(x)=\langle f,K(\cdot,x)\rangle_{\mathcal H}.
$$

The kernel “reproduces” evaluation. In particular,

$$
|f(x)|\le\|f\|_{\mathcal H}\sqrt{K(x,x)}.
$$

This is the link between controlling a function's RKHS norm and controlling its predictions. Ordinary $L^2$ need not be an RKHS: changing a function at one point does not change its $L^2$ element, so point evaluation is not generally well-defined.

## 3. From a kernel back to its RKHS

_Lectures 18-19; [pp. 68-71](615.pdf#page=68)._

The **Moore-Aronszajn theorem** states that every positive semidefinite kernel determines a unique RKHS with that reproducing kernel.

Start from finite linear combinations

$$
f=\sum_i a_iK(\cdot,x_i),\qquad g=\sum_jb_jK(\cdot,z_j),
$$

and define

$$
\langle f,g\rangle=\sum_{i,j}a_ib_jK(x_i,z_j).
$$

Positive semidefiniteness makes squared norms nonnegative. Identify any zero-norm representations, then complete this inner-product space. Evaluation is bounded because

$$
|f(x)|=|\langle f,K(\cdot,x)\rangle|
\le\|f\|\sqrt{K(x,x)},
$$

so it extends continuously to the completion. This produces the RKHS.

This construction makes precise the infinite-sum notation in the PDF: membership and convergence are governed by the RKHS norm. An arbitrary formal infinite sum is not automatically a member.

### Example 1: the Gaussian kernel

For $K(x,z)=e^{-\|x-z\|^2}$, each Gaussian bump $K(\cdot,x_i)$ lies in the RKHS, as does any finite sum. For example,

$$
f(\cdot)=K(\cdot,x_1)+2K(\cdot,x_2)
$$

has squared norm

$$
\|f\|_{\mathcal H}^2
=K(x_1,x_1)+4K(x_1,x_2)+4K(x_2,x_2).
$$

The coefficients and pairwise kernel values determine its norm.

### Example 2: derive the kernel $\min(s,t)$

Consider absolutely continuous functions on $[0,1]$ with $f(0)=0$ and $f'\in L^2[0,1]$, with

$$
\langle f,g\rangle_{\mathcal H}=\int_0^1f'(t)g'(t)\,dt.
$$

The derivative map identifies this space with $L^2$, so it is complete. Also

$$
|f(x)|=\left|\int_0^x f'(t)\,dt\right|
\le\sqrt{x}\left(\int_0^x|f'(t)|^2\,dt\right)^{1/2}
\le\|f\|_{\mathcal H}.
$$

Thus evaluation is continuous. To find its representer, rewrite

$$
f(x)=\int_0^1 f'(t)\mathbf1_{[0,x]}(t)\,dt.
$$

We need $\partial_tK(t,x)=\mathbf1_{[0,x]}(t)$ almost everywhere and $K(0,x)=0$. Integrating,

$$
K(t,x)=\int_0^t\mathbf1_{[0,x]}(s)\,ds=\min(t,x).
$$

Substituting its derivative into the inner product verifies the reproducing identity.

## 4. Regularized ERM and the representer theorem

_Lectures 19-20; [pp. 71-74](615.pdf#page=71)._

Let $\mathcal H$ be the RKHS of $K$. Consider

$$
\min_{f\in\mathcal H}
\frac1n\sum_i\ell(f(X_i),Y_i)+\lambda\|f\|_{\mathcal H}^2,
\qquad \lambda>0.
$$

The PDF uses summed loss plus $C\|f\|^2$; dividing its objective by $n$ gives $\lambda=C/n$. This $C$ multiplies the norm, unlike the soft-margin convention in Part 3 where $C$ multiplies the loss.

Since $f(X_i)=\langle f,K(\cdot,X_i)\rangle_{\mathcal H}$, squared-loss fitting is ridge regression on the canonical features $\Phi(X_i)=K(\cdot,X_i)$. But $\mathcal H$ may be infinite-dimensional.

### Representer theorem and proof

If a minimizer exists, every minimizer has the form

$$
\hat f(x)=\sum_{i=1}^n a_iK(x,X_i).
$$

Let $S=\operatorname{span}\{K(\cdot,X_1),\ldots,K(\cdot,X_n)\}$ and decompose

$$
f=f_S+f_\perp,\qquad f_S\in S,\quad f_\perp\perp S.
$$

Because $K(\cdot,X_i)\in S$,

$$
f_\perp(X_i)=\langle f_\perp,K(\cdot,X_i)\rangle_{\mathcal H}=0.
$$

Therefore $f(X_i)=f_S(X_i)$ for every training input: their empirical losses are identical. Pythagoras gives

$$
\|f\|_{\mathcal H}^2=\|f_S\|_{\mathcal H}^2+\|f_\perp\|_{\mathcal H}^2.
$$

If $f_\perp\ne0$, removing it strictly reduces the penalty because $\lambda>0$. Hence no minimizer can have a nonzero perpendicular component, proving the theorem. This argument does not require convex loss, though convexity helps solve the resulting problem.

### The finite coefficient problem

Set $G_{ij}=K(X_i,X_j)$ and $a=(a_1,\ldots,a_n)^\top$. Then

$$
\begin{aligned}
(\hat f(X_1),\ldots,\hat f(X_n))^\top&=Ga,\\
\|\hat f\|_{\mathcal H}^2
&=\sum_{i,j}a_ia_j\langle K(\cdot,X_i),K(\cdot,X_j)\rangle_{\mathcal H}\\
&=a^\top Ga.
\end{aligned}
$$

The optimization therefore becomes

$$
\min_{a\in\mathbb R^n}
\frac1n\sum_i\ell((Ga)_i,Y_i)+\lambda a^\top Ga.
$$

Infinite-dimensional learning has reduced to at most $n$ coefficients.

## 5. Kernel regression and classification losses

_Lectures 19-20; [pp. 73-76](615.pdf#page=73)._

### Kernel ridge regression

For squared loss, minimize

$$
J(a)=\frac1n\|Ga-y\|^2+\lambda a^\top Ga.
$$

Since $G$ is symmetric,

$$
\nabla J(a)=\frac2nG(Ga-y)+2\lambda Ga.
$$

Setting the gradient to zero gives

$$
G[(G+n\lambda I)a-y]=0.
$$

A solution is obtained by solving the linear system

$$
(G+n\lambda I)a=y.
$$

This matrix is positive definite: for $v\ne0$,

$$
v^\top(G+n\lambda I)v=v^\top Gv+n\lambda\|v\|^2>0.
$$

Thus $a=(G+n\lambda I)^{-1}y$ is valid even if $G$ is singular. It satisfies the stationarity equation, and convexity makes it optimal. Predict using $\hat f(x)=\sum_i a_iK(x,X_i)$.

**Clarification of p. 73:** the formula $(G^2+CG)^{-1}Gy$ given there assumes invertible $G$ and simplifies to $(G+CI)^{-1}y$. The latter remains valid for singular $G$. Coefficients can then have additional null-space representations, but the fitted RKHS function is unique.

### Quantile regression

For $\tau\in(0,1)$, define the pinball loss

$$
\ell_\tau(u)=
\begin{cases}
(\tau-1)u,&u<0,\\
\tau u,&u\ge0.
\end{cases}
$$

Why does it estimate a quantile? For fixed $x$, consider $Q(a)=\mathbb E[\ell_\tau(Y-a)\mid X=x]$, assuming integrability. For $Y<a$, the derivative with respect to $a$ is $1-\tau$; for $Y>a$, it is $-\tau$. At continuity points of the conditional distribution,

$$
Q'(a)=(1-\tau)\Pr(Y<a\mid x)-\tau\Pr(Y>a\mid x)
=F_{Y\mid x}(a)-\tau.
$$

Thus the minimum occurs at the conditional $\tau$-quantile. With atoms, the subgradient condition becomes $F_{Y\mid x}(a-)\le\tau\le F_{Y\mid x}(a)$.

Its regularized sample version is

$$
\min_{f\in\mathcal H}\frac1n\sum_i\ell_\tau(Y_i-f(X_i))
+\lambda\|f\|_{\mathcal H}^2.
$$

The representer theorem applies as before.

### Kernel logistic regression

For $Y_i\in\{-1,1\}$, use

$$
\ell(s,Y_i)=-\log\frac{e^{Y_is}}{1+e^{Y_is}}
=\log(1+e^{-Y_is}).
$$

Minimize the regularized objective to obtain a real score $\hat f(x)$. Convert it to a hard prediction by its sign, or report the model probability

$$
\hat p(x)=\frac{e^{\hat f(x)}}{1+e^{\hat f(x)}}.
$$

The lecture also describes drawing a random class using this probability. That is a randomized prediction rule, distinct from reporting the probability or thresholding it.

### Kernel hinge loss and SVMs

For $\ell(s,y)=\max\{0,1-ys\}$, regularized ERM becomes

$$
\min_{f\in\mathcal H}
\frac1n\sum_i\max\{0,1-Y_if(X_i)\}+\lambda\|f\|_{\mathcal H}^2.
$$

Multiplying by $1/(2\lambda)$ gives norm penalty $\|f\|^2/2$ and loss coefficient $C=1/(2n\lambda)$, matching the soft-margin formulation. An unpenalized intercept can be added to match Part 3 exactly.

Squared, quantile, logistic, and hinge losses are convex in the score. Since $Ga$ is linear in $a$ and $G$ is positive semidefinite, all these coefficient problems are convex.

## 6. Stationary kernels and Bochner's theorem

_Lecture 20; [pp. 76-77](615.pdf#page=76)._

A **stationary** kernel depends only on displacement:

$$
K(x,z)=h(x-z).
$$

For a random frequency vector $\omega$, its characteristic function is

$$
\varphi_\omega(u)=\mathbb E[e^{i\omega^\top u}].
$$

If $\omega$ and $-\omega$ have the same distribution, the imaginary sine terms cancel, giving $\varphi_\omega(u)=\mathbb E\cos(\omega^\top u)$.

To see positive semidefiniteness, for real coefficients $c_i$,

$$
\begin{aligned}
\sum_{i,j}c_ic_j\varphi_\omega(x_i-x_j)
&=\mathbb E\sum_{i,j}c_ic_j
e^{i\omega^\top x_i}e^{-i\omega^\top x_j}\\
&=\mathbb E\left|\sum_i c_i e^{i\omega^\top x_i}\right|^2\ge0.
\end{aligned}
$$

**Bochner's theorem** gives the converse for continuous stationary positive semidefinite kernels: each nonzero real such kernel can be written

$$
K(x,z)=A\,\mathbb E\cos(\omega^\top(x-z)),
\qquad A=K(0,0)>0,
$$

for a symmetric frequency distribution. The zero kernel is the trivial separate case. Continuity is needed in this statement.

### The lecture's examples

The rational quadratic family is

$$
K(x,z)=\left(1+\frac{\|x-z\|^2}{\alpha\ell^2}\right)^{-\alpha},
\qquad\alpha,\ell>0.
$$

For $r=\|x-z\|^2/\ell^2$, take logarithms: $-\alpha\log(1+r/\alpha)\to-r$. Hence this family approaches $e^{-\|x-z\|^2/\ell^2}$ as $\alpha\to\infty$.

Another family is

$$
K(x,z)=\exp\left(-\frac{\|x-z\|^\gamma}{\ell^\gamma}\right),
\qquad0<\gamma\le2.
$$

At $\gamma=2$ it is Gaussian; at $\gamma=1$ it is exponential, corresponding to a Cauchy-type spectral distribution. The PDF also points to the Matérn family as further kernel examples, without deriving its formula.

## 7. Gaussian processes and covariance kernels

_Lecture 21; [pp. 78-81](615.pdf#page=78)._

A **Gaussian process** is a collection of random variables indexed by inputs, $\{Z(x):x\in\mathcal X\}$, such that every finite collection is jointly Gaussian. For a zero-mean process with covariance function $K$,

$$
(Z(x_1),\ldots,Z(x_m))^\top\sim\mathcal N(0,G),
\qquad G_{ij}=K(x_i,x_j).
$$

A sample is therefore a random function. The covariance describes how its values vary together across inputs.

Covariance functions are kernels. Symmetry is immediate, and

$$
\sum_{i,j}c_ic_jK(x_i,x_j)
=\operatorname{Var}\left(\sum_i c_iZ(x_i)\right)\ge0.
$$

Conversely, a positive semidefinite kernel defines compatible Gaussian distributions on finite input sets, and hence a Gaussian process. The lectures develop a spectral construction to explain how to sample one.

## 8. Mercer expansion: the spectral view of a kernel

_Lecture 21; [pp. 79-80](615.pdf#page=79)._

Take a bounded continuous kernel on $\mathbb R^d$ and a strictly positive probability density $p$. Define an integral operator on $L^2(p)$:

$$
(Tf)(x)=\int K(x,z)f(z)p(z)\,dz.
$$

This generalizes matrix-vector multiplication: the sum over a matrix column becomes an integral over $z$. It is an integral operator, not generally a convolution.

### The operator is bounded

If $|K(x,z)|\le M$, Cauchy-Schwarz gives

$$
|(Tf)(x)|^2
\le\left(\int K(x,z)^2p(z)\,dz\right)
\left(\int f(z)^2p(z)\,dz\right)
\le M^2\|f\|_{L^2(p)}^2.
$$

Integrating over $x$ and using $\int p=1$ yields

$$
\|Tf\|_{L^2(p)}\le M\|f\|_{L^2(p)}.
$$

Symmetry of $K$ makes $T$ self-adjoint; positive semidefiniteness makes it positive. Square integrability of $K$ also makes it compact. These are the operator counterparts of the finite-dimensional spectral setting.

### Eigenfunctions replace eigenvectors

A symmetric positive semidefinite matrix has an expansion $G=\sum_k\lambda_k u_ku_k^\top$. The corresponding Mercer expansion is

$$
T\psi_k=\lambda_k\psi_k,\qquad
\langle\psi_k,\psi_l\rangle_{L^2(p)}=\delta_{kl},
\qquad
K(x,z)=\sum_k\lambda_k\psi_k(x)\psi_k(z).
$$

The positive eigenvalues tend to zero if there are infinitely many. Zero-eigenvalue directions may be needed to complete a basis of $L^2(p)$, but contribute nothing to the kernel expansion. This avoids assuming that the positive eigenfunctions alone span the whole space.

The spectral series converges in $L^2(p\otimes p)$; the continuous representatives supplied by the Mercer setting give the pointwise kernel expansion, locally uniformly. The density $p$ determines the eigenfunctions and eigenvalues, even though the represented kernel is fixed.

### Derive the coefficients

For fixed $x$, boundedness gives $K(\cdot,x)\in L^2(p)$. Expand it in the spectral basis:

$$
K(\cdot,x)=\sum_k\langle K(\cdot,x),\psi_k\rangle_{L^2(p)}\psi_k(\cdot).
$$

Compute each coefficient:

$$
\begin{aligned}
\langle K(\cdot,x),\psi_k\rangle_{L^2(p)}
&=\int K(z,x)\psi_k(z)p(z)\,dz\\
&=(T\psi_k)(x)=\lambda_k\psi_k(x).
\end{aligned}
$$

Substitution yields the Mercer expansion. This calculation uses the $L^2(p)$ inner product; the RKHS reproducing inner product is a different one.

The expansion also supplies an explicit feature map:

$$
\Phi(x)=(\sqrt{\lambda_1}\psi_1(x),\sqrt{\lambda_2}\psi_2(x),\ldots),
\qquad
\langle\Phi(x),\Phi(z)\rangle_{\ell^2}=K(x,z).
$$

## 9. Karhunen-Loève expansion and sampling

_Lectures 21-22; [pp. 80-83](615.pdf#page=80)._

Let $\xi_k$ be independent $\mathcal N(0,1)$ variables. Define

$$
Z(x)=\sum_k\sqrt{\lambda_k}\,\xi_k\psi_k(x).
$$

For a finite truncation, any finite set of values is a linear transformation of independent Gaussians, so it is jointly Gaussian. Moreover,

$$
\mathbb E\left|\sum_{k>M}\sqrt{\lambda_k}\xi_k\psi_k(x)\right|^2
=\sum_{k>M}\lambda_k\psi_k(x)^2\to0,
$$

because the full diagonal sum equals $K(x,x)<\infty$. The finite-dimensional Gaussian vectors therefore have a mean-square limit.

The process has mean zero. Its covariance is

$$
\begin{aligned}
\operatorname{Cov}(Z(x),Z(z))
&=\sum_{k,l}\sqrt{\lambda_k\lambda_l}\,
\psi_k(x)\psi_l(z)\mathbb E[\xi_k\xi_l]\\
&=\sum_k\lambda_k\psi_k(x)\psi_k(z)\\
&=K(x,z).
\end{aligned}
$$

Only terms with $k=l$ survive because $\mathbb E[\xi_k\xi_l]=\delta_{kl}$. This proves that the expansion produces the desired GP.

### Method 1: truncate the expansion

Draw $M$ standard normals and use

$$
Z_M(x)=\sum_{k=1}^M\sqrt{\lambda_k}\xi_k\psi_k(x).
$$

The sampled function can be evaluated at any input. The cost is needing the eigenpairs and accepting the omitted spectral tail.

### Method 2: sample on a finite input grid

Choose $x_1,\ldots,x_m$, form $G_{ij}=K(x_i,x_j)$, and sample $\mathcal N(0,G)$ directly. For example, if $G=U\Lambda U^\top$, draw $\xi\sim\mathcal N(0,I)$ and set

$$
z=U\Lambda^{1/2}\xi.
$$

Then $\mathbb Ez=0$ and $\operatorname{Cov}(z)=U\Lambda U^\top=G$. This also works for singular $G$. If $G$ is positive definite, a Cholesky factor $G=LL^\top$ gives $z=L\xi$.

This method needs no continuous-domain eigenfunctions. It produces values at the chosen grid points; interpolation is an additional approximation.

## 10. Work out the eigenpairs of $\min(s,t)$

_Lecture 22; [pp. 83-84](615.pdf#page=83)._

On $[0,1]$ with uniform measure,

$$
(Tf)(t)=\int_0^1\min(s,t)f(s)\,ds.
$$

For a positive eigenvalue $\lambda$, split the eigenvalue equation at $s=t$:

$$
\lambda f(t)=\int_0^t s f(s)\,ds+t\int_t^1f(s)\,ds.
$$

Differentiate using the fundamental theorem of calculus and the product rule:

$$
\lambda f'(t)=tf(t)+\int_t^1f(s)\,ds-tf(t)
=\int_t^1f(s)\,ds.
$$

Differentiate again:

$$
\lambda f''(t)=-f(t).
$$

The original equation at $t=0$ gives $f(0)=0$, while the first derivative at $t=1$ gives $f'(1)=0$.

Set $\omega=1/\sqrt{\lambda}$. The differential equation has solution

$$
f(t)=A\sin(\omega t)+B\cos(\omega t).
$$

The first boundary condition forces $B=0$. The second gives $A\omega\cos\omega=0$. A nonzero eigenfunction requires

$$
\cos\omega=0
\quad\Longrightarrow\quad
\omega=(k-\tfrac12)\pi,\qquad k=1,2,\ldots.
$$

Finally,

$$
\int_0^1\sin^2((k-\tfrac12)\pi t)\,dt=\frac12,
$$

so normalization gives

$$
\boxed{
\lambda_k=\frac1{(k-\tfrac12)^2\pi^2},\qquad
\psi_k(t)=\sqrt2\sin((k-\tfrac12)\pi t).
}
$$

**Correction to p. 84:** the printed integer frequencies $k\pi$ do not satisfy $f'(1)=0$. The half-integer frequencies above follow from the lecture's own boundary conditions.

Substituting these eigenpairs into the preceding expansion gives a GP with covariance $\min(s,t)$, namely standard Brownian motion on $[0,1]$.

## 11. Why the course looks beyond fixed kernels

_Lecture 22; [pp. 84-85](615.pdf#page=84)._

Kernel learning with convex losses offers tractable convex objectives. Gaussian kernels also have a **universal approximation** property: on compact $A\subset\mathbb R^d$, their RKHS is dense in the continuous functions under the uniform norm.

This means that for continuous $g$ and every $\varepsilon>0$, some RKHS function satisfies

$$
\sup_{x\in A}|f(x)-g(x)|\le\varepsilon.
$$

Existence says nothing by itself about how many basis functions or observations efficient learning needs.

The lecture illustrates the **curse of dimensionality** by the worst-case approximation scaling for a bounded smoothness class:

$$
n\text{ on the order of }\varepsilon^{-d/k}
$$

for $k$ derivatives of smoothness. A grid interpretation is that local error of order $h^k$ needs $h\approx\varepsilon^{1/k}$, while covering a $d$-dimensional region needs about $h^{-d}$ cells.

For $k=2,d=400,\varepsilon=1/2$, the illustration becomes $2^{200}$. This is a worst-case approximation-complexity example, not a universal sample-size formula for every kernel or target.

Structure can change the problem. A compositional function may have the form

$$
g(x_1,\ldots,x_d)=h\big(h(x_1,x_2),h(x_3,x_4)\big)
$$

with further pairwise composition for larger $d$. A deep network can represent that hierarchy directly, motivating the discussion of deep learning.

The PDF uses summation as a simple compositional example. Summation is also easy for a linear kernel, so this example alone does not establish a neural-network advantage. Efficiency depends on the target structure and the chosen model. Universal approximation is also distinct from universal statistical consistency.

## 12. Conformal prediction

_Lecture 23; [pp. 87-90](615.pdf#page=87)._

So far a model predicts a value, label, or probability. We now ask for a set $C(x)$ whose coverage has a finite-sample guarantee.

### Step 1: fit using training data

Use any fitting method to obtain a predictor. For classification it can output real class scores; it need not output a hard label.

Keep a separate calibration sample $(\widetilde X_i,\widetilde Y_i)_{i=1}^m$. Fix both predictor and scoring rule before using it. Under the lecture's assumption, training, calibration, and future test pairs are i.i.d.

### Step 2: define a nonconformity score

Choose $s(x,y)$ to be larger when outcome $y$ agrees less with the prediction. The lecture examples are:

$$
\begin{array}{ll}
\text{Binary, }y\in\{-1,1\}:&s(x,y)=e^{-yf(x)},\\[1mm]
\text{Multiclass, scores }f_j(x):&s(x,y)=e^{-f_y(x)}.
\end{array}
$$

For a strongly positive binary score, label $+1$ has a small nonconformity score and label $-1$ has a large one. The score need not be a calibrated probability.

### Step 3: calibrate a threshold

Compute $s_i=s(\widetilde X_i,\widetilde Y_i)$ and sort

$$
s_{(1)}\le\cdots\le s_{(m)}.
$$

For target error level $\alpha\in(0,1)$, define

$$
k=\lceil(m+1)(1-\alpha)\rceil,\qquad
q=\begin{cases}
s_{(k)},&k\le m,\\
+\infty,&k=m+1.
\end{cases}
$$

The $m+1$ adjustment accounts for the future score as one more exchangeable observation. The infinite-threshold case completes the definition when the requested coverage exceeds what a finite calibration quantile can supply.

For $m=99$ and $\alpha=0.1$, $k=90$, so use the 90th smallest calibration score.

### Step 4: invert the score

Output

$$
C(x)=\{y:s(x,y)\le q\}.
$$

In classification, test each candidate class against the same threshold. Depending on scores, the set can be empty, contain one label, or contain several. Larger sets express less decisive predictions.

### Regression with estimated quantiles

Fit lower and upper quantile functions $L(x),U(x)$, for example at levels $0.25$ and $0.75$ as in the PDF. Assume the initial bounds are ordered. Define

$$
s(x,y)=\max\{L(x)-y,\ y-U(x),\ 0\}.
$$

Because these scores are nonnegative, the calibrated $q$ is nonnegative. The condition $s(x,y)\le q$ is equivalent to the three inequalities

$$
L(x)-y\le q,\qquad y-U(x)\le q,\qquad0\le q.
$$

Rearranging the first two yields

$$
C(x)=[L(x)-q,\ U(x)+q].
$$

Calibration therefore expands the initial interval by a common amount sufficient for the desired marginal coverage.

### Derive the coverage guarantee

Condition on the fitted predictor and fixed score rule. The calibration scores and future true-label score

$$
s_{\mathrm{new}}=s(X_{\mathrm{new}},Y_{\mathrm{new}})
$$

are exchangeable. If there are no ties, the future score's rank among all $m+1$ scores is uniform on $\{1,\ldots,m+1\}$.

For $k\le m$, $s_{\mathrm{new}}\le s_{(k)}$ exactly when that rank is at most $k$. Hence

$$
\Pr(s_{\mathrm{new}}\le q)=\frac{k}{m+1}\ge1-\alpha.
$$

With ties, imagine random tie-breaking only for the rank argument. Every rank-at-most-$k$ event is still accepted by the rule using $\le$, so ties can only increase coverage. If $k=m+1$, the threshold is infinite and coverage is 1.

Since membership in $C(X_{\mathrm{new}})$ is precisely $s_{\mathrm{new}}\le q$,

$$
\Pr\{Y_{\mathrm{new}}\in C(X_{\mathrm{new}})\}\ge1-\alpha.
$$

This is **marginal coverage**: it averages over calibration data and the new pair, conditional on model fitting under the i.i.d. setup. It is not a $1-\alpha$ guarantee for every individual input or for every fixed calibration set.

Using calibration labels to fit or tune the score breaks this simple exchangeability argument. A poor model can still obtain valid coverage by producing large sets; coverage and informativeness are separate properties.
