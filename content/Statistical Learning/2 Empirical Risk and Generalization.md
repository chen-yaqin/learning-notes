---
title: 2. Empirical Risk and Generalization
description: STAT615 lectures 9-13 — ERM, uniform convergence, VC bounds, and Rademacher symmetrization.
aliases:
  - Empirical Risk Minimization
  - VC Dimension and Rademacher Complexity
tags:
  - statistical-learning
  - stat615
---

[[1 Prediction and Consistency|Previous: prediction and consistency]] · [[3 Linear Classifiers and Support Vector Machines|Next: linear classifiers and SVMs]]

Source: [Lectures 9-13, pp. 31-46](615.pdf#page=31).

Part 1 estimated conditional probabilities and then classified. We now choose a predictor with small observed loss. The difficulty is that we select the predictor using the same sample on which we measure its error.

## 1. Why unrestricted training-error minimization fails

Under 0-1 loss,

$$
R_n(f)=\frac1n\sum_i\mathbf1\{f(X_i)\ne Y_i\},
\qquad R(f)=\Pr(f(X)\ne Y).
$$

Although Bayes minimizes $R$, minimizing $R_n$ over all functions can overfit.

Suppose $\rho_X$ has a density. Surround the distinct training inputs by disjoint balls whose union $B$ has probability at most $\varepsilon$. Define

$$
f_n(x)=\begin{cases}
Y_i,&x\text{ lies in the ball around }X_i,\\
1-f_B(x),&x\notin B.
\end{cases}
$$

Every label is memorized, so $R_n(f_n)=0$. Outside $B$, the conditional error is $\max\{\eta(x),1-\eta(x)\}\ge1/2$. Thus

$$
R(f_n)\ge\frac12\rho_X(B^c)\ge\frac{1-\varepsilon}{2}\ge\frac12-\varepsilon.
$$

This is a counterexample, not an implementable algorithm: using the unknown Bayes rule shows that low training error alone does not identify a good predictor.

## 2. Restrict to a hypothesis class

Choose a fixed class $\mathcal F$, such as affine half-space classifiers. Distinguish

$$
\hat f_n\in\arg\min_{f\in\mathcal F}R_n(f),
\qquad
f^*_{\mathcal F}\in\arg\min_{f\in\mathcal F}R(f),
\qquad
f_B\in\arg\min_fR(f).
$$

These are the best training fit, best population fit within the class, and unrestricted optimum. We write minimizers when they exist; infima and approximate minimizers give analogous bounds with an optimization-error term.

We want to compare $R(\hat f_n)$ with its training error, with $R(f^*_{\mathcal F})$, and with Bayes risk $R^*$.

### Uniform convergence

Define

$$
\Delta_n=\sup_{f\in\mathcal F}|R_n(f)-R(f)|.
$$

First, $R(\hat f_n)-R_n(\hat f_n)\le\Delta_n$. For the second comparison, add and subtract empirical risks:

$$
\begin{aligned}
R(\hat f_n)-R(f^*_{\mathcal F})
&=[R(\hat f_n)-R_n(\hat f_n)]\\
&\quad+[R_n(\hat f_n)-R_n(f^*_{\mathcal F})]\\
&\quad+[R_n(f^*_{\mathcal F})-R(f^*_{\mathcal F})]\\
&\le\Delta_n+0+\Delta_n=2\Delta_n.
\end{aligned}
$$

The middle term is nonpositive because $\hat f_n$ minimizes training risk.

A bound for one fixed $f$ is insufficient: choosing $\hat f_n$ after seeing the data changes the independence argument. A bound uniform over a predetermined class covers the selected predictor too.

### Approximation error remains

$$
\begin{aligned}
R(\hat f_n)-R^*
&=[R(\hat f_n)-R(f^*_{\mathcal F})]+[R(f^*_{\mathcal F})-R^*]\\
&\le2\Delta_n+
\underbrace{\inf_{f\in\mathcal F}R(f)-R^*}_{\operatorname{Approx}(\mathcal F)}.
\end{aligned}
$$

The first term controls estimation error. The second measures how well the class represents an optimal rule. A richer class can reduce approximation error while making estimation harder.

**Clarification of the lecture's consistency statements:** uniform convergence implies consistency relative to the best predictor in $\mathcal F$. Bayes consistency additionally requires approximation error to vanish. Necessity statements in VC learning theory involve distribution-free quantifiers; a pointwise claim for one distribution is not the full equivalence.

## 3. Finite classes: Hoeffding and a union bound

For fixed $f$, the losses are independent and bounded in $[0,1]$, so

$$
\Pr(|R_n(f)-R(f)|\ge t)\le2e^{-2nt^2}.
$$

If $\mathcal F=\{f_1,\ldots,f_M\}$, a union bound gives

$$
\Pr(\Delta_n\ge t)
\le\sum_{j=1}^M\Pr(|R_n(f_j)-R(f_j)|\ge t)
\le2Me^{-2nt^2}.
$$

Set $2Me^{-2nt^2}=\delta$ and solve:

$$
t=\sqrt{\frac{\log(2M/\delta)}{2n}}.
$$

With probability at least $1-\delta$, this bounds $\Delta_n$, and twice this quantity bounds the ERM excess risk within the class.

For infinite classes, cardinality is too crude: different functions can behave identically on a finite sample.

## 4. Shattering and the growth function

_Lectures 10-11; [pp. 36-39](615.pdf#page=36)._

For inputs $x_1,\ldots,x_n$, define

$$
\mathcal F_{x_1,\ldots,x_n}
=\{(f(x_1),\ldots,f(x_n)):f\in\mathcal F\}.
$$

The **growth function**, or shattering number, is

$$
S(n,\mathcal F)=\sup_{x_1,\ldots,x_n}|\mathcal F_{x_1,\ldots,x_n}|\le2^n.
$$

A set is **shattered** when every binary labeling can be realized. The supremum ranges over arrangements; one unshatterable arrangement does not establish a bound on $S$.

### Affine classifiers in the plane

- One point admits both labels: $S(1)=2$.
- Two distinct points admit all four labelings: $S(2)=4$.
- Three collinear points cannot realize $(1,0,1)$ in their linear order. Three noncollinear points can realize all eight labelings, so $S(3)=8$.
- Four points cannot be shattered. Alternating labels on a convex quadrilateral cannot be separated by a line. If one point lies in the convex hull of the others, it cannot be separated from all the others. Degenerate arrangements also fail.

Thus affine half-spaces in $\mathbb R^2$ have VC dimension 3.

### Deriving the VC probability bound

The lecture states, for $t\ge\sqrt{2/n}$,

$$
\Pr(\Delta_n\ge t)\le4S(2n,\mathcal F)e^{-nt^2/8}.
$$

The following steps explain the independent-copy argument behind it.

**1. Introduce a ghost sample.** Let $D'_n$ be independent, with empirical risk $R'_n$. For a function selected using $D_n$, its ghost risk has conditional variance at most $1/(4n)$. Chebyshev gives

$$
\Pr(|R'_n(f)-R(f)|>t/2\mid D_n)\le\frac1{nt^2}\le\frac12.
$$

If a training sample exhibits deviation at least $t$, with probability at least $1/2$ the ghost risk is within $t/2$ of population risk. The two empirical risks then differ by at least $t/2$. With the usual measurability and supremum-approximation details,

$$
\Pr(\Delta_n\ge t)
\le2\Pr\left(\sup_f|R_n(f)-R'_n(f)|\ge t/2\right).
$$

**2. Randomly swap paired observations.** Swapping $(X_i,Y_i)$ with $(X'_i,Y'_i)$ independently leaves the joint distribution unchanged. Conditional on the pooled observations, the difference becomes

$$
\frac1n\sum_i\sigma_i d_i(f),
\qquad
d_i(f)=\ell(f(X_i),Y_i)-\ell(f(X'_i),Y'_i)\in[-1,1],
$$

where $\sigma_i$ are independent random signs. Each summand has range width at most 2. Hoeffding bounds the probability of deviation $t/2$ for one labeling by $2e^{-nt^2/8}$.

**3. Count behaviors.** At most $S(2n,\mathcal F)$ label vectors occur on the pooled inputs. A union bound contributes this factor; Step 1 contributes another factor 2. This gives the displayed VC inequality.

Solving for failure probability $\delta$ yields

$$
\Delta_n\le\sqrt{\frac8n\log\frac{4S(2n,\mathcal F)}{\delta}}
$$

with probability at least $1-\delta$. Exponential growth of $S(2n)$ need not give a shrinking bound, so we next control that growth.

## 5. VC dimension controls growth

Define

$$
v=\operatorname{VC}(\mathcal F)=\sup\{m:S(m,\mathcal F)=2^m\}.
$$

The **Sauer-Shelah lemma**, stated in the PDF, gives

$$
S(n,\mathcal F)\le\sum_{j=0}^{\min(v,n)}\binom nj.
$$

For $1\le v\le n$, this is at most $(en/v)^v$. The restriction $v\le n$ matters; this expression is not a bound for arbitrary smaller $n$.

Substitute $S(2n)\le(2en/v)^v$ into the probability bound:

$$
\Delta_n\le
\sqrt{\frac8n\left[v\log\frac{2en}{v}+\log\frac4\delta\right]}
\quad(1\le v\le2n).
$$

For fixed finite $v$, $\log n/n\to0$ makes the bound vanish. Taking $\delta_n=n^{-2}$ makes failure probabilities summable; Borel-Cantelli then gives almost-sure uniform convergence.

### Examples and empirical distribution functions

Affine half-spaces in $\mathbb R^d$ have VC dimension $d+1$. Thresholds $f_t(x)=\mathbf1\{x\le t\}$ have VC dimension 1: one point is shattered, but two ordered points cannot receive labels $(0,1)$.

Applied to threshold averages, uniform convergence says

$$
\sup_t\left|\frac1n\sum_i\mathbf1\{X_i\le t\}-F_X(t)\right|\to0
\quad\text{almost surely}.
$$

This is the Glivenko-Cantelli property. To express it literally as a classification risk, take the constant label $Y=0$, so the threshold indicator equals its loss.

VC dimension measures worst-case input arrangements. Rademacher complexity uses the observed inputs or their distribution.

## 6. Rademacher complexity

_Lectures 11-13; [pp. 39-46](615.pdf#page=39)._

Switch to labels and classifiers in $\{-1,1\}$. Independently of the sample, draw $\sigma_i$ with equally likely values $-1$ and $1$. Define

$$
\widehat{\mathfrak R}_n(\mathcal F)
=\mathbb E_\sigma\left[\sup_{f\in\mathcal F}\frac1n\sum_i\sigma_i f(X_i)\right],
\qquad
\mathfrak R_n(\mathcal F)=\mathbb E_X\widehat{\mathfrak R}_n(\mathcal F).
$$

Empirical complexity measures how well the class can align with random labels on the observed inputs.

For a singleton $\{f_0\}$, linearity and $\mathbb E\sigma_i=0$ give complexity 0. If the class shatters the distinct sample inputs, choose $f(X_i)=\sigma_i$ to get

$$
\sup_f\frac1n\sum_i\sigma_i f(X_i)=\frac1n\sum_i\sigma_i^2=1.
$$

No correlation exceeds 1, so empirical complexity is exactly 1 in this case. Repeated inputs cannot always fit contradictory random labels.

### Step 1: symmetrization with a ghost sample

To handle absolute values correctly, define two signed suprema:

$$
G_+=\sup_f(R(f)-R_n(f)),\qquad
G_-=\sup_f(R_n(f)-R(f)).
$$

Then $\Delta_n=\max\{G_+,G_-\}$. Since $R(f)=\mathbb E_{D'_n}R'_n(f)$, moving the supremum outside the ghost expectation gives

$$
\mathbb E G_+\le\mathbb E_{D_n,D'_n}\sup_f[R'_n(f)-R_n(f)].
$$

For our label convention,

$$
\mathbf1\{f(X)\ne Y\}=\frac{1-Yf(X)}2.
$$

Consequently,

$$
R'_n(f)-R_n(f)=\frac1{2n}\sum_i[Y_i f(X_i)-Y'_i f(X'_i)].
$$

### Step 2: introduce random signs

Swapping each observation with its ghost partner independently leaves the joint law unchanged. Therefore

$$
\begin{aligned}
\mathbb E G_+
&\le\frac12\mathbb E\sup_f\frac1n\sum_i
\sigma_i[Y_i f(X_i)-Y'_i f(X'_i)]\\
&\le\frac12\mathbb E\sup_f\frac1n\sum_i\sigma_iY_i f(X_i)\\
&\quad+\frac12\mathbb E\sup_f\frac1n\sum_i(-\sigma_i)Y'_i f(X'_i).
\end{aligned}
$$

The last inequality uses $\sup_f(A_f+B_f)\le\sup_fA_f+\sup_fB_f$.

### Step 3: absorb the labels

Conditional on all inputs and labels, multiplying an independent symmetric sign by a fixed $Y_i\in\{-1,1\}$ leaves it symmetric. The signs remain independent. Both expectations above thus equal $\mathfrak R_n$, giving

$$
\mathbb E G_+\le\mathfrak R_n(\mathcal F).
$$

Reversing the two samples proves $\mathbb E G_-\le\mathfrak R_n(\mathcal F)$.

**Correction to pp. 43-46:** symmetry of the signs alone does not allow removal of an absolute value inside a supremum. A singleton has signed complexity zero but need not have zero expected absolute random correlation. The two signed-supremum argument avoids that step.

### Step 4: concentration around expectation

Replacing one training observation changes $R_n(f)$ by at most $1/n$ for every $f$. Since

$$
|\sup_fa_f-\sup_fb_f|\le\sup_f|a_f-b_f|,
$$

each $G_\pm$ has bounded differences $c_i=1/n$. McDiarmid gives

$$
\Pr(G_\pm-\mathbb EG_\pm\ge t)\le e^{-2nt^2}.
$$

Union-bound the two events and take $t=\sqrt{\log(2/\delta)/(2n)}$. With probability at least $1-\delta$,

$$
\Delta_n\le\mathfrak R_n(\mathcal F)+\sqrt{\frac{\log(2/\delta)}{2n}}.
$$

These constants follow from the stated $\{-1,1\}$ convention; the PDF displays looser constants.

### Step 5: use empirical complexity

Replacing one input changes a random correlation by at most $2/n$. Taking suprema and sign expectations preserves this bound, so $\widehat{\mathfrak R}_n$ has bounded differences $2/n$. Its lower-tail bound is

$$
\Pr(\mathfrak R_n-\widehat{\mathfrak R}_n\ge2t)\le e^{-2nt^2}.
$$

Combine this with the two events from Step 4. Set $t=\sqrt{\log(3/\delta)/(2n)}$ so the total failure probability is at most $\delta$. Then

$$
\Delta_n\le\widehat{\mathfrak R}_n(\mathcal F)
+3\sqrt{\frac{\log(3/\delta)}{2n}}.
$$

The extra term pays for estimating distribution-dependent complexity using the observed inputs.

## 7. Estimation in practice and the conclusion

For the fixed inputs:

1. Draw $B$ independent random-label vectors $\sigma^{(1)},\ldots,\sigma^{(B)}$.
2. Solve $m_b=\sup_f n^{-1}\sum_i\sigma_i^{(b)}f(X_i)$ for each vector.
3. Approximate empirical complexity by $B^{-1}\sum_bm_b$.

This is ERM with random labels because

$$
\frac1n\sum_i\mathbf1\{f(X_i)\ne\sigma_i\}
=\frac12-\frac1{2n}\sum_i\sigma_i f(X_i).
$$

Generating labels is cheap; optimizing can be expensive. A heuristic optimizer can underestimate the supremum, so it does not by itself provide a certified complexity upper bound.

If $\mathfrak R_n\to0$ for a given distribution, summable failure probabilities imply $\Delta_n\to0$ almost surely. ERM reaches the best risk within $\mathcal F$. For a universal statement, the condition must hold for every distribution. For Bayes consistency, approximation error must also vanish.

Local methods estimate $\eta$ and use consistency arguments; ERM methods control uniform deviations through class complexity. We now choose concrete classes and objectives: [[3 Linear Classifiers and Support Vector Machines|continue to Part 3]].
