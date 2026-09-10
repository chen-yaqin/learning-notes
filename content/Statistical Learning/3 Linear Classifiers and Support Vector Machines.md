---
title: 3. Linear Classifiers and Support Vector Machines
description: STAT615 lectures 13-17 — Gaussian discriminants, logistic regression, margin geometry, convex duality, and kernelization.
aliases:
  - Linear Classifiers
  - Support Vector Machines
tags:
  - statistical-learning
  - stat615
---

[[2 Empirical Risk and Generalization|Previous: empirical risk and generalization]] · [[4 Kernels Gaussian Processes and Conformal Prediction|Next: kernels, Gaussian processes, and conformal prediction]]

Source: [Lectures 13-17, pp. 47-63](615.pdf#page=47).

The previous part explained how controlling a class can support generalization. We now choose concrete classifiers. LDA and logistic regression can both produce linear boundaries, but they estimate different quantities. SVMs choose boundaries through their geometry.

## 1. Geometry of a linear classifier

Use binary labels $Y\in\{-1,1\}$ and a score $s(x)=w^\top x+b$, with $w\ne0$. Predict $+1$ for $s(x)\ge0$ and $-1$ otherwise. The boundary is

$$
H_{w,b}=\{x:w^\top x+b=0\}.
$$

If $x_0,x\in H_{w,b}$, then

$$
w^\top(x-x_0)=(-b)-(-b)=0.
$$

Thus $w$ is perpendicular to the hyperplane. Projecting $x-x_0$ onto its unit normal gives

$$
\operatorname{dist}(x,H_{w,b})
=\left|\left\langle x-x_0,\frac w{\|w\|}\right\rangle\right|
=\frac{|w^\top x+b|}{\|w\|}.
$$

The two half-spaces correspond to positive and negative scores. Positive rescaling does not change the boundary or classifier:

$$
H_{cw,cb}=H_{w,b}\quad(c>0).
$$

We will use this scaling freedom to turn geometric margin maximization into a tractable optimization problem.

## 2. QDA and LDA from the Bayes rule

_Lectures 13-14; [pp. 48-51](615.pdf#page=48)._

For classes $j=1,\ldots,K$, assume

$$
\Pr(Y=j)=\pi_j,\qquad X\mid Y=j\sim\mathcal N(\mu_j,\Sigma_j),
$$

with positive priors and positive-definite covariances. Bayes' rule gives

$$
f_B(x)=\arg\max_j\Pr(Y=j\mid X=x)
=\arg\max_j\pi_jp_j(x).
$$

The density is

$$
p_j(x)=\frac{\exp[-\tfrac12(x-\mu_j)^\top\Sigma_j^{-1}(x-\mu_j)]}
{(2\pi)^{d/2}\det(\Sigma_j)^{1/2}}.
$$

Taking logarithms preserves the ordering. Dropping the common term $-\tfrac d2\log(2\pi)$, choose the largest score

$$
d_j(x)=\log\pi_j-\frac12\log\det\Sigma_j
-\frac12(x-\mu_j)^\top\Sigma_j^{-1}(x-\mu_j).
$$

### QDA: separate covariances

Expand the quadratic form:

$$
(x-\mu_j)^\top\Sigma_j^{-1}(x-\mu_j)
=x^\top\Sigma_j^{-1}x
-2x^\top\Sigma_j^{-1}\mu_j
+\mu_j^\top\Sigma_j^{-1}\mu_j.
$$

When $\Sigma_j$ depends on the class, the quadratic term remains in $d_j-d_l$. The pairwise boundaries $d_j(x)=d_l(x)$ are generally quadratic: **quadratic discriminant analysis**.

With $n_j=\#\{i:Y_i=j\}$, estimate

$$
\hat\pi_j=\frac{n_j}{n},\qquad
\hat\mu_j=\frac1{n_j}\sum_{i:Y_i=j}X_i,
$$

$$
\hat\Sigma_j=\frac1{n_j}\sum_{i:Y_i=j}
(X_i-\hat\mu_j)(X_i-\hat\mu_j)^\top.
$$

These are the maximum-likelihood estimates under the model. Substitute them into $d_j$ to obtain the fitted classifier. The displayed inverse formulas require nonsingular estimates.

### LDA: a shared covariance

Suppose every class has covariance $\Sigma$. Then

$$
\begin{aligned}
d_j(x)
&=\underbrace{-\frac12\log\det\Sigma-\frac12x^\top\Sigma^{-1}x}_{\text{common to all classes}}\\
&\quad+x^\top\Sigma^{-1}\mu_j
-\frac12\mu_j^\top\Sigma^{-1}\mu_j+\log\pi_j.
\end{aligned}
$$

Common terms do not affect the maximum. Therefore compare the affine scores

$$
\ell_j(x)=x^\top\Sigma^{-1}\mu_j
-\frac12\mu_j^\top\Sigma^{-1}\mu_j+\log\pi_j.
$$

Their pairwise differences are linear in $x$: **linear discriminant analysis**. Estimate the common covariance by pooling within-class covariances, for example the MLE

$$
\hat\Sigma=\frac1n\sum_j\sum_{i:Y_i=j}
(X_i-\hat\mu_j)(X_i-\hat\mu_j)^\top.
$$

The Gaussian model explains the rule's derivation. The resulting plug-in classifier can still be used when the true distribution is not Gaussian, but then its Bayes optimality is not guaranteed.

## 3. Logistic regression from conditional probabilities

_Lecture 14; [p. 52](615.pdf#page=52)._

Logistic regression models $Y\mid X$ directly. It does not specify a Gaussian model for $X\mid Y$.

For $K$ classes, set $s_j(x)=\beta_j^\top x+b_j$ for $j<K$ and use the reference score $s_K(x)=0$. Define

$$
p_j(x)=\frac{e^{s_j(x)}}{1+\sum_{l=1}^{K-1}e^{s_l(x)}}\quad(j<K),
\qquad
p_K(x)=\frac1{1+\sum_{l=1}^{K-1}e^{s_l(x)}}.
$$

The shared denominator cancels in an argmax. Exponentiation preserves order, so

$$
\arg\max_jp_j(x)=\arg\max_js_j(x).
$$

Thus pairwise decision boundaries are affine despite the nonlinear probabilities.

### Fit by maximum likelihood

For independent labeled observations, maximize the conditional likelihood $\prod_i p_{Y_i}(X_i)$. Taking logs and changing the sign gives

$$
\min_{\{\beta_j,b_j\}}
\sum_i\left[
\log\left(1+\sum_{l<K}e^{s_l(X_i)}\right)-s_{Y_i}(X_i)
\right],
$$

where $s_K=0$. This is empirical minimization of negative log-likelihood.

For binary labels $Y_i\in\{-1,1\}$ and one score $s(x)$,

$$
\Pr(Y=y\mid X=x)=\frac1{1+e^{-ys(x)}},
$$

so the objective becomes

$$
\min_{w,b}\frac1n\sum_i\log(1+e^{-Y_i(w^\top X_i+b)}).
$$

A large positive signed score $Y_is(X_i)$ means a confident correct prediction and a small loss. A negative signed score receives a larger penalty.

## 4. From perceptron motivation to maximum margin

_Lectures 14-15; [pp. 52-55](615.pdf#page=52)._

The PDF motivates the perceptron by penalizing distances of wrongly classified observations to the boundary. For $w\ne0$, wrong-side distances can be written as

$$
\sum_{i:Y_i(w^\top X_i+b)<0}
\frac{-Y_i(w^\top X_i+b)}{\|w\|}.
$$

This motivates correcting wrong-side points, but does not distinguish all the separators of a separable dataset. It also gives no positive margin requirement.

Assume both classes occur and the data are **strictly linearly separable**:

$$
Y_i(w^\top X_i+b)>0\quad\text{for every }i
$$

for some $w,b$. Strict positivity matters; classification by a tie convention at score zero does not ensure a positive margin.

### Derive the hard-margin problem

The margin of a separating hyperplane is its smallest training-point distance:

$$
\operatorname{margin}(w,b)
=\min_i\frac{Y_i(w^\top X_i+b)}{\|w\|}.
$$

Let $m=\min_iY_i(w^\top X_i+b)>0$. Rescale $(w,b)$ by $1/m$. The classifier is unchanged and its minimum signed score becomes 1. For the rescaled parameters,

$$
\operatorname{margin}(w,b)=\frac1{\|w\|}.
$$

Maximizing $1/\|w\|$ is equivalent to minimizing $\|w\|^2/2$. Hence the hard-margin SVM solves

$$
\min_{w,b}\frac12\|w\|^2
\quad\text{subject to}\quad
Y_i(w^\top X_i+b)\ge1\quad\forall i.
$$

The objective is convex and the constraints are affine. At an optimum, the minimum signed score is 1: otherwise scaling both parameters down would preserve feasibility and reduce the objective.

The distance from the decision boundary to the nearest observations is $1/\|w\|$; the distance between the two supporting hyperplanes is $2/\|w\|$.

## 5. Convex duality and KKT

_Lecture 15; [pp. 55-56](615.pdf#page=55)._

Consider a differentiable convex objective $F(z)$ and differentiable convex constraints $h_i(z)\le0$:

$$
\min_z F(z)\quad\text{subject to }h_i(z)\le0.
$$

Introduce multipliers $\alpha_i\ge0$ and the Lagrangian

$$
L(z,\alpha)=F(z)+\sum_i\alpha_i h_i(z).
$$

Define the dual function $g(\alpha)=\inf_zL(z,\alpha)$ and maximize it over $\alpha\ge0$.

### Why the dual gives a lower bound

For any feasible $z$ and $\alpha\ge0$,

$$
g(\alpha)\le L(z,\alpha)
=F(z)+\sum_i\alpha_i h_i(z)\le F(z).
$$

Thus the best dual value cannot exceed the best primal value: **weak duality**.

The primal can also be written as $\inf_z\sup_{\alpha\ge0}L(z,\alpha)$. For feasible $z$, the inner supremum equals $F(z)$; if any constraint is violated, its multiplier can grow without bound. The dual reverses the optimization order. **Strong duality** means the optimal values coincide.

### Slater and the KKT conditions

Slater's condition asks for a strictly feasible point $\tilde z$ with $h_i(\tilde z)<0$ for every inequality. Under the stated convex setting and an attained finite optimum, it ensures the usual strong-duality and multiplier-existence result.

The **Karush-Kuhn-Tucker conditions** are

$$
\begin{aligned}
\nabla F(z^*)+\sum_i\alpha_i^*\nabla h_i(z^*)&=0
&&\text{stationarity},\\
h_i(z^*)&\le0&&\text{primal feasibility},\\
\alpha_i^*&\ge0&&\text{dual feasibility},\\
\alpha_i^*h_i(z^*)&=0&&\text{complementary slackness}.
\end{aligned}
$$

To see sufficiency, stationarity minimizes the convex function $L(\cdot,\alpha^*)$, so $g(\alpha^*)=L(z^*,\alpha^*)$. Complementary slackness makes this equal $F(z^*)$. Weak duality then shows both points are optimal.

For strictly separable SVM data, scale a separator until every signed score is strictly greater than 1. This supplies Slater's point.

## 6. Derive the hard-margin SVM dual

_Lectures 15-16; [pp. 56-60](615.pdf#page=56)._

The constraints are $h_i(w,b)=1-Y_i(w^\top X_i+b)\le0$. Therefore

$$
\begin{aligned}
L(w,b,\alpha)
&=\frac12\|w\|^2+\sum_i\alpha_i[1-Y_i(w^\top X_i+b)]\\
&=\frac12\|w\|^2+\sum_i\alpha_i
-w^\top\sum_i\alpha_iY_iX_i
-b\sum_i\alpha_iY_i.
\end{aligned}
$$

### First minimize over the intercept

If $\sum_i\alpha_iY_i\ne0$, letting $b$ move in the appropriate direction drives the Lagrangian to $-\infty$. A finite dual value thus requires

$$
\sum_i\alpha_iY_i=0.
$$

Under this constraint, the intercept term vanishes.

### Then minimize over the weight vector

Set its gradient to zero:

$$
\nabla_w L=w-\sum_i\alpha_iY_iX_i=0
\quad\Longrightarrow\quad w=\sum_i\alpha_iY_iX_i.
$$

Writing $v=\sum_i\alpha_iY_iX_i$, the $w$ terms at the minimum are $\|v\|^2/2-v^\top v=-\|v\|^2/2$. Hence

$$
g(\alpha)=\sum_i\alpha_i
-\frac12\sum_{i,j}\alpha_i\alpha_jY_iY_jX_i^\top X_j.
$$

The dual is

$$
\max_{\alpha\ge0}
\left[\sum_i\alpha_i-\frac12\sum_{i,j}
\alpha_i\alpha_jY_iY_jX_i^\top X_j\right]
\quad\text{subject to }\sum_i\alpha_iY_i=0.
$$

The matrix $Q_{ij}=Y_iY_jX_i^\top X_j$ is positive semidefinite, since

$$
c^\top Qc=\left\|\sum_i c_iY_iX_i\right\|^2\ge0.
$$

The dual objective is therefore concave.

### Recover the primal solution

Stationarity gives $w^*=\sum_i\alpha_i^*Y_iX_i$. Complementary slackness gives

$$
\alpha_i^*[1-Y_i((w^*)^\top X_i+b^*)]=0.
$$

If $\alpha_i^*>0$, the constraint is tight. Since $Y_i^2=1$,

$$
Y_i((w^*)^\top X_i+b^*)=1
\quad\Longrightarrow\quad
b^*=Y_i-(w^*)^\top X_i.
$$

Such an index exists when both classes are present: if every multiplier were zero, stationarity would give $w^*=0$, and one intercept could not meet the constraints for both signs.

### Support vectors and perturbations

Points with positive multipliers contribute to $w^*$ and lie on the hard margin. The PDF calls every margin point a support vector; in a degenerate solution a margin point may have multiplier zero, so the reverse implication need not hold.

Suppose a point lies strictly outside the margin on the correct side. Its multiplier is zero. Move it while keeping the old separator feasible. The original weights, intercept, and multipliers still satisfy KKT:

- Its zero multiplier leaves stationarity unchanged.
- Feasibility holds by the condition on its new location.
- Its complementary-slackness product remains zero.
- The other constraints and multipliers are unchanged.

Therefore the original solution remains optimal. This explains the limited sensitivity to observations that do not determine the margin.

### Dimensions and inner products

The primal optimizes $d+1$ parameters; the dual optimizes $n$ multipliers. Training and prediction depend on inputs only through inner products:

$$
s(x)=\sum_i\alpha_i^*Y_iX_i^\top x+b^*.
$$

This can be useful when $d$ is large, although dual computation also depends on the cost of forming and storing the $n\times n$ Gram matrix.

## 7. Soft margins and hinge loss

_Lecture 17; [p. 61](615.pdf#page=61)._

Hard-margin constraints are infeasible for nonseparable data. Introduce slack variables:

$$
\min_{w,b,\xi\ge0}\frac12\|w\|^2+C\sum_i\xi_i
\quad\text{subject to}\quad
Y_i(w^\top X_i+b)\ge1-\xi_i.
$$

For fixed $w,b$, each constraint requires both $\xi_i\ge0$ and $\xi_i\ge1-Y_is(X_i)$. Since $C>0$, the smallest feasible slack is optimal:

$$
\xi_i=\max\{0,1-Y_is(X_i)\}.
$$

Substituting gives the equivalent unconstrained objective

$$
\min_{w,b}\frac12\|w\|^2+C\sum_i\max\{0,1-Y_i(w^\top X_i+b)\}.
$$

This is regularization plus **hinge loss**. A correctly classified observation can still incur loss when its signed score is between 0 and 1.

Larger $C$ penalizes violations more strongly; smaller $C$ puts more emphasis on a small norm. The hard-margin limit as $C$ grows requires separable data.

### The corresponding dual constraint

To continue the earlier derivation, add multipliers $\mu_i\ge0$ for $-\xi_i\le0$. The coefficient of $\xi_i$ in the Lagrangian is $C-\alpha_i-\mu_i$. Minimization requires it to vanish:

$$
C-\alpha_i-\mu_i=0
\quad\Longrightarrow\quad0\le\alpha_i\le C.
$$

The other stationarity equations are unchanged. Thus the same dual objective now has box constraints $0\le\alpha_i\le C$ and $\sum_i\alpha_iY_i=0$.

If $0<\alpha_i<C$, then $\mu_i>0$ forces $\xi_i=0$, and complementary slackness gives $Y_is(X_i)=1$. Such points can recover the intercept. A soft-margin support vector with $\alpha_i=C$ may lie inside the margin or be misclassified.

### Choose the tuning parameter

Fit the model for each candidate $C$ on training data. Evaluate its classification error on a separate validation set and choose the value with the smallest validation error. Cross-validation repeats this fit-and-evaluate procedure over folds. Keep final test data separate from tuning.

## 8. Nonlinear geometry and the kernel trick

_Lecture 17; [pp. 61-63](615.pdf#page=61)._

A linear boundary cannot separate every useful geometry. For points in inner and outer rings in $\mathbb R^2$, the map

$$
\Phi(x_1,x_2)=(x_1,x_2,x_1^2+x_2^2)
$$

adds squared radius as a coordinate. A plane in this feature space can separate radii that no line in the original plane can separate.

For a general feature map $\Phi:\mathcal X\to\mathcal H$, the hard-margin problem becomes

$$
\min_{w,b}\frac12\|w\|_{\mathcal H}^2
\quad\text{subject to}\quad
Y_i(\langle w,\Phi(X_i)\rangle_{\mathcal H}+b)\ge1.
$$

The same stationarity calculation gives

$$
w^*=\sum_i\alpha_i^*Y_i\Phi(X_i).
$$

Thus the dual uses only $\langle\Phi(X_i),\Phi(X_j)\rangle_{\mathcal H}$. Define

$$
K(x,z)=\langle\Phi(x),\Phi(z)\rangle_{\mathcal H}.
$$

Replace every original inner product by $K$. Prediction becomes

$$
s(x)=\sum_i\alpha_i^*Y_iK(X_i,x)+b^*.
$$

For a hard-margin point with positive multiplier,

$$
b^*=Y_i-\sum_j\alpha_j^*Y_jK(X_j,X_i).
$$

This rule is linear in the feature space and can be nonlinear in the original inputs. We can compute it without explicitly constructing $\Phi$.

The next question is which functions $K$ really behave as feature-space inner products. That leads to kernels and RKHS: [[4 Kernels Gaussian Processes and Conformal Prediction|continue to Part 4]].
