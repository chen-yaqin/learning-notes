---
title: 1. Prediction and Consistency
description: STAT615 lectures 0-9 — prediction, Bayes classifiers, local averaging, concentration, and consistency.
aliases:
  - Risk and Bayes Classifiers
  - Plug-in Classifiers and Consistency
  - Concentration Inequalities
tags:
  - statistical-learning
  - stat615
---

[[Statistical Learning/index|Course contents]] · [[2 Empirical Risk and Generalization|Next: empirical risk and generalization]]

Source: [Lectures 0-9, pp. 6-31](615.pdf#page=6).

## 1. Prediction, loss, and risk

_Lecture 0; [pp. 6-8](615.pdf#page=6)._

We observe training data $D_n=\{(X_i,Y_i)\}_{i=1}^n$ and want to predict $Y$ after observing a new input $X$. Assume the training pairs and test pair are independent draws from the same unknown distribution $\rho$ on $\mathcal X\times\mathcal Y$.

This assumption connects past observations to future predictions. One input can have several possible outcomes: its label need not be deterministic.

- **Classification:** $\mathcal Y$ is finite, such as digit labels or traffic signs.
- **Regression:** $\mathcal Y=\mathbb R$ or another continuous output space.
- **Uncertainty quantification:** output probabilities or a set of plausible outcomes.

A loss $\ell(a,y)$ measures the cost of predicting $a$ when the outcome is $y$. The population risk is

$$
R_\ell(f)=\mathbb E_{(X,Y)\sim\rho}[\ell(f(X),Y)].
$$

For a fitted predictor, $R_\ell(\hat f_n)=\mathbb E[\ell(\hat f_n(X),Y)\mid D_n]$ is random through the training data. An outer expectation $\mathbb E_{D_n}$ averages over that randomness.

### Linear regression as the first example

For $f_\theta(x)=\beta^\top x+b$, let $A$ have rows $(X_i^\top,1)$ and let $\theta=(\beta^\top,b)^\top$. Least squares solves

$$
\min_\theta\frac1n\|y-A\theta\|^2.
$$

Differentiating gives

$$
\frac2n A^\top(A\hat\theta-y)=0
\quad\Longrightarrow\quad
A^\top A\hat\theta=A^\top y.
$$

If $A^\top A$ is invertible, $\hat\theta=(A^\top A)^{-1}A^\top y$. Otherwise solve the normal equations with a suitable generalized inverse. Predict with $\hat f_n(x)=\hat\beta^\top x+\hat b$.

Why squared loss? Write $m(x)=\mathbb E[Y\mid X=x]$. For finite conditional second moments,

$$
\begin{aligned}
\mathbb E[(Y-a)^2\mid X=x]
&=\mathbb E[(Y-m(x)+m(x)-a)^2\mid X=x]\\
&=\operatorname{Var}(Y\mid X=x)+(m(x)-a)^2.
\end{aligned}
$$

The cross term vanishes because $\mathbb E[Y-m(x)\mid X=x]=0$. Thus the unrestricted population minimizer is $m(x)$. Restricting to linear functions introduces an approximation choice.

The statistical issue remains: minimizing observed average loss need not minimize its population expectation. This becomes the ERM question in Part 2.

### Other outputs

Logistic regression uses $p(x)=e^{\beta^\top x+b}/(1+e^{\beta^\top x+b})$ and predicts 1 when $p(x)\ge1/2$. Part 3 explains how its parameters are fitted.

Quantile regression estimates a conditional quantile. Two such estimates can form an interval $[L(x),U(x)]$. Estimated probabilities or quantiles alone do not automatically provide a coverage guarantee; Part 4 develops conformal calibration.

## 2. The Bayes classifier

_Lectures 1-2; [pp. 9-13](615.pdf#page=9)._

Under **0-1 loss**, $\ell(a,y)=\mathbf1\{a\ne y\}$, risk is

$$
R(f)=\Pr(f(X)\ne Y).
$$

For binary labels, set $\eta(x)=\Pr(Y=1\mid X=x)$. By the tower property,

$$
R(f)=\mathbb E_X\!\left[\mathbb E[\mathbf1\{f(X)\ne Y\}\mid X]\right].
$$

At a fixed input, the two conditional errors are

$$
\Pr(1\ne Y\mid X=x)=1-\eta(x),
\qquad
\Pr(0\ne Y\mid X=x)=\eta(x).
$$

Choose 1 exactly when $1-\eta(x)\le\eta(x)$, or $\eta(x)\ge1/2$. Therefore

$$
f_B(x)=\mathbf1\{\eta(x)\ge1/2\}.
$$

This minimizes conditional risk at every input; averaging proves $R(f_B)\le R(f)$ for every classifier. The **Bayes risk** is

$$
R^*=R(f_B)=\mathbb E[\min\{\eta(X),1-\eta(X)\}]\le\frac12.
$$

If $\eta(x)=0.8$, even the optimal prediction has conditional error $0.2$.

### Multiclass and unequal costs

For classes $1,\ldots,K$, predicting $j$ has error $1-\Pr(Y=j\mid X=x)$. Choose the largest conditional probability.

For a general cost function, choose

$$
f_B(x)\in\arg\min_a\sum_j\ell(a,j)\Pr(Y=j\mid X=x).
$$

Confusing a stop sign with a speed-limit sign can cost more than confusing two speed limits. The optimal rule changes with the loss.

### Bayes' rule and the lecture examples

With class prior $\pi_j$ and class-conditional density $p_j$,

$$
\Pr(Y=j\mid X=x)=\frac{\pi_jp_j(x)}{\sum_l\pi_lp_l(x)}.
$$

The denominator is common to every class, so compare $\pi_jp_j(x)$ directly. For the PDF's discrete example:

| Input    | $a$ | $b$ | $c$ | $d$ | $e$ | $f$ |
| -------- | --- | --- | --- | --- | --- | --- |
| $q(x,0)$ | 0   | 0.1 | 0.2 | 0.5 | 0.8 | 0.9 |
| $q(x,1)$ | 0.9 | 0.3 | 0.1 | 0.3 | 0.1 | 0.3 |
| $f_B(x)$ | 1   | 1   | 0   | 0   | 0   | 0   |

These are unnormalized joint scores. Normalization cancels, so each column selects its larger score.

For $X\mid Y=j\sim\mathcal N(\mu_j,\sigma_j^2)$, predicting 1 means

$$
\frac{\pi_1}{\sqrt{2\pi\sigma_1^2}}
e^{-(x-\mu_1)^2/(2\sigma_1^2)}
\ge
\frac{\pi_0}{\sqrt{2\pi\sigma_0^2}}
e^{-(x-\mu_0)^2/(2\sigma_0^2)}.
$$

Taking logs and collecting terms gives

$$
\log\frac{\pi_1}{\pi_0}-\frac12\log\frac{\sigma_1^2}{\sigma_0^2}
-\frac{(x-\mu_1)^2}{2\sigma_1^2}
+\frac{(x-\mu_0)^2}{2\sigma_0^2}\ge0.
$$

Different variances produce a quadratic boundary. Equal variances cancel the $x^2$ terms, leaving a linear boundary. This leads to QDA and LDA in Part 3.

## 3. Plug-in classifiers and excess risk

_Lecture 3; [pp. 14-15](615.pdf#page=14)._

Estimate the unknown $\eta$ with $\hat\eta_n$ and set

$$
\hat f_n(x)=\mathbf1\{\hat\eta_n(x)\ge1/2\}.
$$

### Step 1: express excess risk pointwise

Compare the conditional errors of $f$ and $f_B$:

- If $f(x)=f_B(x)$, the difference is zero.
- If $f(x)=1,f_B(x)=0$, it is $(1-\eta(x))-\eta(x)=1-2\eta(x)$.
- If $f(x)=0,f_B(x)=1$, it is $\eta(x)-(1-\eta(x))=2\eta(x)-1$.

Thus

$$
R(f)-R^*=\mathbb E_X\!\left[|2\eta(X)-1|\,
\mathbf1\{f(X)\ne f_B(X)\}\right].
$$

Disagreement matters most where one class is clearly favored.

### Step 2: bound disagreement by estimation error

Let $f(x)=\mathbf1\{a(x)\ge1/2\}$. Disagreement puts $a(x)$ and $\eta(x)$ on opposite sides of $1/2$:

$$
\begin{aligned}
\eta(x)<1/2\le a(x)&\Longrightarrow1-2\eta(x)\le2(a(x)-\eta(x)),\\
a(x)<1/2\le\eta(x)&\Longrightarrow2\eta(x)-1\le2(\eta(x)-a(x)).
\end{aligned}
$$

When the rules agree, the excess-risk integrand is zero. Therefore

$$
0\le R(f)-R^*\le2\mathbb E_X|\eta(X)-a(X)|.
$$

Apply this conditionally on $D_n$ with $a=\hat\eta_n$, then average over training data:

$$
\mathbb E_{D_n}[R(\hat f_n)]-R^*
\le2\mathbb E_{D_n,X}|\hat\eta_n(X)-\eta(X)|.
$$

Vanishing average absolute estimation error implies convergence to Bayes risk. It does not require agreement with Bayes at every input.

## 4. Local averages and consistency

_Lectures 3-5; [pp. 15-20](615.pdf#page=15)._

### Notions of consistency

For a fixed distribution:

$$
\begin{array}{ll}
\text{Consistency in expectation:}&\mathbb E_{D_n}[R(\hat f_n)]\to R^*,\\[2mm]
\text{Strong consistency:}&R(\hat f_n)\to R^*\quad\text{almost surely}.
\end{array}
$$

**Universal** means the corresponding property holds for every distribution on the stated space. Since $0\le R(\hat f_n)-R^*\le1$, dominated convergence shows that strong consistency implies consistency in expectation.

The lecture analogy is mean estimation: the sample mean converges almost surely for every distribution with finite first absolute moment. A sample median estimates a median and is not universally consistent for the mean.

### Weighted label estimators

On $\mathbb R^d$, consider

$$
\hat\eta_n(x)=\sum_iw_{ni}(x)Y_i,
\qquad w_{ni}(x)\ge0,\qquad\sum_iw_{ni}(x)=1.
$$

Weights may depend on $x,X_1,\ldots,X_n$, but not on the labels.

**Kernel smoothing:** with bandwidth $\gamma_n>0$,

$$
w_{ni}(x)=\frac{e^{-\|x-X_i\|^2/\gamma_n^2}}
{\sum_j e^{-\|x-X_j\|^2/\gamma_n^2}}.
$$

**Histograms:** partition space into cubes of side length $h_n$. Let $A_n(x)$ contain $x$ and $N_n(x)=\sum_i\mathbf1\{X_i\in A_n(x)\}$. Use

$$
w_{ni}(x)=\begin{cases}
\mathbf1\{X_i\in A_n(x)\}/N_n(x),&N_n(x)>0,\\
1/n,&N_n(x)=0.
\end{cases}
$$

**Nearest neighbors:** give weight $1/k_n$ to each of the $k_n$ nearest observations and zero to the rest. Resolve distance ties without using labels.

### Stone's theorem

Suppose the normalized, nonnegative weights satisfy:

1. **Stability:** for a constant $C$ and every nonnegative integrable $g$,

   $$
   \mathbb E\sum_iw_{ni}(X)g(X_i)\le C\mathbb E[g(X)].
   $$

2. **Localization:** for every $r>0$,

   $$
   \mathbb E\sum_iw_{ni}(X)\mathbf1\{\|X-X_i\|>r\}\to0.
   $$

3. **Vanishing maximum weight:** $\mathbb E[\max_iw_{ni}(X)]\to0$.

Then $\mathbb E|\hat\eta_n(X)-\eta(X)|\to0$, so the classifiers are universally consistent.

### Derivation: separate smoothing error from label noise

Define $\bar\eta_n(x)=\sum_iw_{ni}(x)\eta(X_i)$. Then

$$
\mathbb E|\eta(X)-\hat\eta_n(X)|
\le\underbrace{\mathbb E|\eta(X)-\bar\eta_n(X)|}_{\text{smoothing error}}
+\underbrace{\mathbb E|\bar\eta_n(X)-\hat\eta_n(X)|}_{\text{label noise}}.
$$

**First suppose $\eta$ is Lipschitz with constant $L$.** Separate neighbors within distance $r$ from those farther away. Since $0\le\eta\le1$,

$$
\begin{aligned}
\mathbb E|\eta(X)-\bar\eta_n(X)|
&\le\mathbb E\sum_iw_{ni}(X)|\eta(X)-\eta(X_i)|\\
&\le Lr+\mathbb E\sum_iw_{ni}(X)\mathbf1\{\|X-X_i\|>r\}.
\end{aligned}
$$

Take $n\to\infty$ by localization, then $r\downarrow0$.

**Now allow measurable $\eta$.** Choose bounded Lipschitz $h_s$ with $\mathbb E|h_s(X)-\eta(X)|\to0$, and let $\bar h_{s,n}(x)=\sum_iw_{ni}(x)h_s(X_i)$. Triangle inequality and stability give

$$
\begin{aligned}
\mathbb E|\eta-\bar\eta_n|
&\le\mathbb E|\eta-h_s|+\mathbb E|h_s-\bar h_{s,n}|
+\mathbb E|\bar h_{s,n}-\bar\eta_n|\\
&\le(1+C)\mathbb E|\eta-h_s|+\mathbb E|h_s-\bar h_{s,n}|.
\end{aligned}
$$

For fixed $s$, the final term vanishes by the Lipschitz argument. Then send $s\to\infty$.

**Finally control label noise.** Conditional on the inputs, $Y_i-\eta(X_i)$ are independent and centered, with variances at most $1/4$. Thus

$$
\begin{aligned}
\mathbb E[(\hat\eta_n(X)-\bar\eta_n(X))^2\mid X,X_1,\ldots,X_n]
&=\sum_iw_{ni}(X)^2\operatorname{Var}(Y_i\mid X_i)\\
&\le\frac14\sum_iw_{ni}(X)^2
\le\frac14\max_iw_{ni}(X).
\end{aligned}
$$

Taking expectations and using Cauchy-Schwarz,

$$
\mathbb E|\hat\eta_n(X)-\bar\eta_n(X)|
\le\frac12\sqrt{\mathbb E\max_iw_{ni}(X)}\to0.
$$

This includes the noise calculation assigned as an exercise in the PDF.

### Applying the conditions

For regular histograms, sufficient conditions are $h_n\to0$ and $nh_n^d\to\infty$: cells shrink while their typical sample counts increase. This interpretation is intuitive; the theorem does not require an input density.

For $k_n$-NN, sufficient conditions are $k_n\to\infty$ and $k_n/n\to0$. The maximum weight $1/k_n$ vanishes, while the decreasing sample fraction makes the neighbor radius shrink at almost every test input. A fixed $k$ does not average away label noise.

The lecture states these applications; verifying their geometric weight conditions is separate from Stone's general proof. To strengthen expected-error convergence to almost-sure convergence, we next need concentration inequalities.

## 5. From limit theorems to concentration

_Lectures 5-6; [pp. 20-23](615.pdf#page=20)._

For i.i.d. $Z_i$ with mean $\mu$, the strong law gives $\bar Z_n\to\mu$ almost surely when $\mathbb E|Z_i|<\infty$. If $0<\sigma^2=\operatorname{Var}(Z_i)<\infty$, the central limit theorem gives

$$
\frac{\sqrt n(\bar Z_n-\mu)}{\sigma}\Rightarrow\mathcal N(0,1).
$$

With $\gamma=\mathbb E|Z_i-\mu|^3<\infty$, Berry-Esseen bounds the normal-approximation error:

$$
\sup_t\left|\Pr\!\left(\frac{\sqrt n(\bar Z_n-\mu)}{\sigma}\le t\right)-\Phi(t)\right|
\le\frac{C\gamma}{\sigma^3\sqrt n}.
$$

**Correction to p. 20:** the third moment is centered and the denominator contains $\sigma^3$. Even this quantitative approximation can be too coarse when the tail probability of interest is smaller than its approximation error. Concentration bounds directly control finite-sample tails.

### Markov and Chernoff

For $W\ge0$ and $t>0$, $t\mathbf1\{W\ge t\}\le W$. Taking expectations proves

$$
\Pr(W\ge t)\le\frac{\mathbb EW}{t}.
$$

More generally, apply Markov to a nonnegative increasing function $\varphi(W)$ with $\varphi(t)>0$:

$$
\Pr(W\ge t)\le\frac{\mathbb E\varphi(W)}{\varphi(t)}.
$$

Taking $\varphi(w)=e^{sw}$, $s>0$, gives for any real $W$

$$
\Pr(W\ge t)\le e^{-st}\mathbb E e^{sW}.
$$

Minimize over $s>0$ to obtain the **Chernoff bound**. Its usefulness depends on controlling the moment-generating function.

### Sub-Gaussian sums

A centered variable $Z$ is sub-Gaussian with parameter $\sigma^2$ if

$$
\mathbb E e^{sZ}\le e^{\sigma^2s^2/2}\quad\text{for every }s\in\mathbb R.
$$

The parameter bounds tail behavior and need not equal the actual variance. Suppose independent centered $Z_i$ have parameters $\sigma_i^2$, and set $W=n^{-1}\sum_i Z_i$. Independence gives

$$
\begin{aligned}
\Pr(W\ge t)
&\le\inf_{s>0}e^{-st}\prod_i\mathbb E e^{sZ_i/n}\\
&\le\inf_{s>0}\exp\!\left(-st+\frac{s^2}{2n^2}\sum_i\sigma_i^2\right).
\end{aligned}
$$

The derivative of the exponent is $-t+s\sum_i\sigma_i^2/n^2$, so its minimizer is $s^*=tn^2/\sum_i\sigma_i^2$. Substituting yields

$$
\Pr(W\ge t)\le\exp\!\left(-\frac{n^2t^2}{2\sum_i\sigma_i^2}\right).
$$

Apply the same argument to $-W$ and add the tail bounds:

$$
\Pr(|W|\ge t)\le2\exp\!\left(-\frac{n^2t^2}{2\sum_i\sigma_i^2}\right).
$$

For a common parameter $\sigma^2$, the exponent is $-nt^2/(2\sigma^2)$.

### Hoeffding's lemma and inequality

If $Z\in[a,b]$, convexity puts the exponential below its endpoint chord:

$$
e^{sz}\le\frac{b-z}{b-a}e^{sa}+\frac{z-a}{b-a}e^{sb}.
$$

The resulting bound on the centered moment-generating function is **Hoeffding's lemma**:

$$
\mathbb E e^{s(Z-\mathbb EZ)}\le e^{s^2(b-a)^2/8}.
$$

One way to finish the chord argument is to use its two-point distribution at $a,b$. Its log moment-generating function $H$ has $H(0)=H'(0)=0$ after centering. The second derivative is a variance under exponential reweighting, bounded by $(b-a)^2/4$. Integrating twice gives $H(s)\le s^2(b-a)^2/8$.

Thus $Z-\mathbb EZ$ is sub-Gaussian with parameter $(b-a)^2/4$. Substitution into the sum bound proves, for independent $Z_i\in[a_i,b_i]$,

$$
\Pr\!\left(\left|\frac1n\sum_i(Z_i-\mathbb EZ_i)\right|\ge t\right)
\le2\exp\!\left(-\frac{2n^2t^2}{\sum_i(b_i-a_i)^2}\right).
$$

The variables need not be identically distributed.

### Bernstein's inequality

Hoeffding uses ranges. Bernstein also uses variance. For independent centered $Z_i$ with $|Z_i|\le M$, let $V=\sum_i\mathbb EZ_i^2$ and $S=\sum_iZ_i$. Then

$$
\Pr(S\ge t)\le\exp\!\left(-\frac{t^2}{2(V+Mt/3)}\right).
$$

The PDF states this result. Its connection to Chernoff can be seen from the following calculation. For $0<s<3/M$, expand the exponential. Since $\mathbb E Z_i=0$, $|Z_i|^k\le M^{k-2}Z_i^2$, and $k!\ge2\cdot3^{k-2}$ for $k\ge2$,

$$
\begin{aligned}
\mathbb E e^{sZ_i}
&\le1+\frac{s^2\mathbb EZ_i^2}{2}\sum_{k=2}^\infty(sM/3)^{k-2}\\
&=1+\frac{s^2\mathbb EZ_i^2}{2(1-sM/3)}\\
&\le\exp\!\left(\frac{s^2\mathbb EZ_i^2}{2(1-sM/3)}\right).
\end{aligned}
$$

Multiply these bounds and apply Chernoff. Choosing $s=t/(V+Mt/3)$ gives the displayed result when $V>0$; $V=0$ is deterministic. Replacing $t$ with $nt$ gives the sample-mean version.

Small variance can improve the bound, but the comparison with Hoeffding also depends on $t$. There is no variance threshold that makes Bernstein uniformly better for all deviations.

## 6. Martingales and bounded differences

_Lecture 7; [pp. 24-25](615.pdf#page=24)._

Our estimator is a function of independent observations, not usually a sum of independent terms. Martingales extend concentration to this setting.

A martingale $M_k$ is integrable, depends on information $\mathcal F_k$ available by step $k$, and satisfies

$$
\mathbb E[M_k\mid\mathcal F_{k-1}]=M_{k-1}.
$$

Its increments $D_k=M_k-M_{k-1}$ have conditional mean zero. Independent centered partial sums are an example.

### Hoeffding-Azuma

Suppose each $D_k$, given the past, lies in an interval of deterministic length $c_k$. Conditional Hoeffding's lemma gives

$$
\mathbb E[e^{sD_k}\mid\mathcal F_{k-1}]\le e^{s^2c_k^2/8}.
$$

Repeated conditioning replaces the independence step used earlier:

$$
\mathbb E e^{s\sum_kD_k}\le e^{s^2\sum_kc_k^2/8}.
$$

Chernoff minimization with $s=4t/\sum_kc_k^2$ gives

$$
\Pr\!\left(\sum_kD_k\ge t\right)
\le\exp\!\left(-\frac{2t^2}{\sum_kc_k^2}\right).
$$

Here $c_k$ is an interval width. If the assumption is instead $|D_k|\le b_k$, use width $2b_k$.

### McDiarmid's inequality

Let $Z_1,\ldots,Z_n$ be independent. Suppose replacing coordinate $i$ changes $g(Z_1,\ldots,Z_n)$ by at most $c_i$:

$$
|g(z_1,\ldots,z_i,\ldots,z_n)-g(z_1,\ldots,z'_i,\ldots,z_n)|\le c_i.
$$

Define the Doob martingale

$$
M_k=\mathbb E[g\mid Z_1,\ldots,Z_k],
\qquad M_0=\mathbb Eg,\quad M_n=g.
$$

Given the first $k-1$ observations, changing $Z_k$ changes the conditional expectation over future observations by at most $c_k$. Thus $D_k=M_k-M_{k-1}$ has conditional range of width at most $c_k$. Since $g-\mathbb Eg=\sum_kD_k$, Hoeffding-Azuma gives

$$
\Pr(g-\mathbb Eg\ge t)\le e^{-2t^2/\sum_i c_i^2},
\qquad
\Pr(|g-\mathbb Eg|\ge t)\le2e^{-2t^2/\sum_i c_i^2}.
$$

For $g=n^{-1}\sum_iZ_i$ with $Z_i\in[a_i,b_i]$, choose $c_i=(b_i-a_i)/n$. This recovers Hoeffding exactly.

## 7. Examples, high probability, and almost-sure convergence

_Lecture 8; [pp. 25-27](615.pdf#page=25)._

### Binomial count

If $N=\sum_iZ_i$ with independent $Z_i\sim\operatorname{Bernoulli}(p)$, then $Z_i\in[0,1]$. The one-sided Hoeffding bound at sample-mean threshold $t/n$ gives

$$
\Pr(N-np\ge t)\le e^{-2n(t/n)^2}=e^{-2t^2/n}.
$$

### Heavy tails and a bounded transformation

For Cauchy observations, the mean is undefined, so the preceding sub-Gaussian and bounded-sum assumptions fail. The density decays like $t^{-2}$; the tail probability decays like $t^{-1}$.

Nevertheless $g(z_1,\ldots,z_n)=\cos(n^{-1}\sum_i z_i)$ changes by at most 2 when any coordinate changes. McDiarmid applies with $c_i=2$:

$$
\Pr(g-\mathbb Eg\ge t)\le e^{-t^2/(2n)}.
$$

The bound is valid but becomes uninformative as $n$ grows. A bounded output alone does not imply useful concentration.

If instead $Z_i\in[a_i,b_i]$, use $|\cos u-\cos v|\le|u-v|$, obtained by integrating the derivative $-\sin$. Then $c_i=(b_i-a_i)/n$, giving the much stronger Hoeffding-sized exponent.

### Convert a tail bound into a confidence statement

Suppose $\Pr(|W_n-\mathbb EW_n|\ge t)\le2e^{-cnt^2}$. Set the right side equal to $\delta$:

$$
\delta=2e^{-cnt^2}
\Longrightarrow\log(2/\delta)=cnt^2
\Longrightarrow t=\sqrt{\frac{\log(2/\delta)}{cn}}.
$$

Thus, with probability at least $1-\delta$, the deviation is at most this threshold.

### Borel-Cantelli

If $\sum_n\Pr(E_n)<\infty$, only finitely many $E_n$ occur almost surely. Independence of the events is not required for this direction.

For example, if $\Pr(Z_n=1)=p_n$ and $\sum_np_n<\infty$, then almost surely only finitely many of the $Z_n$ equal 1.

To prove convergence of bounded sample means, suppose $|Z_i|\le M$, the observations are independent with common mean $\mu$, and

$$
\Pr(|\bar Z_n-\mu|\ge t)\le2e^{-nt^2/(2M^2)}.
$$

Choose $t_n=\sqrt{4M^2\log n/n}$ for $n\ge2$. Then $t_n\to0$ and

$$
\sum_{n=2}^\infty\Pr(|\bar Z_n-\mu|\ge t_n)
\le2\sum_{n=2}^\infty n^{-2}<\infty.
$$

Borel-Cantelli implies $|\bar Z_n-\mu|<t_n$ eventually almost surely, proving convergence. This proves the strong law in the bounded setting; the general strong law is broader.

## 8. Strong consistency of histogram classifiers

_Lectures 8-9; [pp. 27-31](615.pdf#page=27)._

We return to the original learning problem. For deterministic regular histogram bins, assume $h_n\to0$ and $nh_n^d\to\infty$. The lecture proves universal strong consistency by replacing a random denominator with a deterministic one for analysis.

### Step 1: rewrite the decision

In a nonempty cell, predicting 1 means

$$
\frac{\sum_iY_i\mathbf1\{X_i\in A_n(x)\}}{N_n(x)}\ge\frac12.
$$

Since $N_n(x)=\sum_i[Y_i+(1-Y_i)]\mathbf1\{X_i\in A_n(x)\}$, this is equivalent to

$$
\sum_iY_i\mathbf1\{X_i\in A_n(x)\}
\ge\sum_i(1-Y_i)\mathbf1\{X_i\in A_n(x)\}.
$$

Let $\mu=\rho_X$. Divide both sides by $n\mu(A_n(x))$ and define

$$
\hat a_n^1(x)=\frac{\sum_iY_i\mathbf1\{X_i\in A_n(x)\}}{n\mu(A_n(x))},
\qquad
\hat a_n^0(x)=\frac{\sum_i(1-Y_i)\mathbf1\{X_i\in A_n(x)\}}{n\mu(A_n(x))}.
$$

Cells of zero $\mu$-mass do not affect risk. In empty cells both scores are zero, so either label is an allowable maximizer, including the earlier fallback. These scores are proof devices: the classifier does not need to know $\mu$.

### Step 2: extend the excess-risk bound

Write $\eta_1=\eta$ and $\eta_0=1-\eta$. If the chosen label $b$ differs from a Bayes label $b^*$, then $\hat a_n^b\ge\hat a_n^{b^*}$. Therefore

$$
\begin{aligned}
\eta_{b^*}-\eta_b
&=(\eta_{b^*}-\hat a_n^{b^*})
+(\hat a_n^{b^*}-\hat a_n^b)
+(\hat a_n^b-\eta_b)\\
&\le|\eta_{b^*}-\hat a_n^{b^*}|+|\hat a_n^b-\eta_b|.
\end{aligned}
$$

When the labels agree the excess is zero. Integrating gives

$$
R(\hat f_n)-R^*
\le\mathbb E_X|\eta(X)-\hat a_n^1(X)|
+\mathbb E_X|1-\eta(X)-\hat a_n^0(X)|.
$$

It suffices to show each term vanishes almost surely.

### Step 3: control the expectation

Focus on $g_n(D_n)=\mathbb E_X|\eta(X)-\hat a_n^1(X)|$. Decompose

$$
g_n=(g_n-\mathbb E g_n)+\mathbb E g_n.
$$

Here is the expectation argument underlying the adaptation mentioned in the PDF. Let $\eta_n^A(x)=\mathbb E[\eta(X)\mid X\in A_n(x)]$. Then

$$
\mathbb E g_n
\le\mathbb E_X|\eta-\eta_n^A|
+\mathbb E_{D_n,X}|\eta_n^A-\hat a_n^1|.
$$

The first term vanishes by approximating $\eta$ with bounded Lipschitz functions: cell averaging contracts $L^1$ error, and a Lipschitz function varies by at most $L\sqrt d\,h_n$ within a cell.

For the second term, in a cell $A$ of mass $p_A>0$, $\hat a_n^1$ is unbiased for $\eta_A$, and

$$
\operatorname{Var}_{D_n}(\hat a_n^1(x))\le\frac1{np_A}
\quad(x\in A).
$$

Thus the cell's integrated expected error is at most $p_A/\sqrt{np_A}=\sqrt{p_A/n}$. In any fixed bounded box, the number $M_n$ of intersecting cells is $O(h_n^{-d})$, so Cauchy-Schwarz gives

$$
\sum_{A\text{ meeting the box}}\sqrt{p_A/n}
\le\sqrt{M_n/n}\to0.
$$

Outside those cells, nonnegativity and unbiasedness bound the integrated error by twice the probability mass outside the box. Take $n\to\infty$, then enlarge the box to make that mass vanish. Therefore $\mathbb E g_n\to0$ without assuming an input density.

### Step 4: concentrate around the expectation

Replace one training pair $(X_i,Y_i)$ by $(X'_i,Y'_i)$. Triangle inequality bounds the change in $g_n$ by

$$
\int\frac{Y_i\mathbf1\{X_i\in A_n(x)\}
+Y'_i\mathbf1\{X'_i\in A_n(x)\}}{n\mu(A_n(x))}\,d\mu(x).
$$

The first integral equals $Y_i/n$: on the cell containing $X_i$, its probability mass cancels the denominator. The second is $Y'_i/n$. Hence the change is at most $2/n$.

McDiarmid now gives

$$
\Pr(|g_n-\mathbb E g_n|\ge t)
\le2\exp\!\left(-\frac{2t^2}{n(2/n)^2}\right)
=2e^{-nt^2/2}.
$$

Choose $t_n=\sqrt{c\log n/n}$ with $c>2$. The failure probabilities are at most $2n^{-c/2}$ and are summable. Borel-Cantelli implies $g_n-\mathbb Eg_n\to0$ almost surely. Together with Step 3, $g_n\to0$ almost surely.

The class-0 term follows by replacing $Y_i$ with $1-Y_i$. Step 2 then proves $R(\hat f_n)\to R^*$ almost surely for every input-label distribution.

We have learned by estimating local probabilities. The next lectures ask whether we can instead minimize observed classification error directly: [[2 Empirical Risk and Generalization|continue to Part 2]].
