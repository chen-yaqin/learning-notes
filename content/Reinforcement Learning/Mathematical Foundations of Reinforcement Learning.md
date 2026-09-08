---
title: Mathematical Foundations of Reinforcement Learning
description: A beginner-friendly map of the mathematical ideas connecting Bellman equations, dynamic programming, Monte Carlo, temporal-difference learning, value approximation, policy gradients, and actor-critic methods.
aliases:
  - Mathematical Foundation of Reinforcement Learning
  - Math Foundations of RL
tags:
  - rl
  - book-notes
date: 2026-09-07
source: https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning
---

These are my notes on Shiyu Zhao's _Mathematical Foundations of Reinforcement Learning_. The book's main strength is that it presents reinforcement learning (RL) as one connected story: define the decision problem, express long-term value recursively, and then turn that recursion into algorithms that can plan or learn from experience.

> [!abstract] The whole book in one sentence
> An agent improves its behavior by repeatedly comparing its current prediction with a better target, then moving the prediction or policy a small step toward that target.

## 1. Describing sequential decisions

### The agent–environment loop

At time $t$:

- $S_t$ is the state;
- $A_t$ is the action selected by the agent;
- $R_{t+1}$ is the reward received afterward; and
- $S_{t+1}$ is the next state.

The loop is therefore

$$
S_t \longrightarrow A_t \longrightarrow (R_{t+1}, S_{t+1}) \longrightarrow A_{t+1} \longrightarrow \cdots
$$

A **policy** tells the agent how to act. A stochastic policy is a conditional probability:

$$
\pi(a \mid s)=\Pr(A_t=a\mid S_t=s).
$$

It can assign all probability to one action (a deterministic policy) or spread probability over several actions.

### Markov decision process

A Markov decision process (MDP) supplies the mathematical model for the loop. Its main ingredients are:

$$
(\mathcal S,\mathcal A,p,r,\gamma),
$$

where:

- $\mathcal S$ is the set of states;
- $\mathcal A$ is the set of actions;
- $p(s'\mid s,a)$ describes the transition dynamics;
- $r(s,a,s')$ is the expected immediate reward; and
- $\gamma\in[0,1)$ is the discount factor.

The **Markov property** means that the current state contains all information needed to predict the next step. Once $S_t$ and $A_t$ are known, earlier states do not add useful predictive information about $S_{t+1}$.

> [!example] Grid-world interpretation
> A square is a state, moving up is an action, the chance of landing in a neighboring square is the transition probability, and reaching the goal may give reward $+1$. A sufficiently informative state must include anything from the past that still matters for the future.

### Reward is immediate; return is long-term

The **discounted return** from time $t$ is

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

The discount factor controls how strongly the future matters:

- a small $\gamma$ makes the agent short-sighted;
- a $\gamma$ close to $1$ makes future rewards important; and
- discounting keeps an infinite sum finite when rewards are bounded.

The useful recursive identity

$$
G_t=R_{t+1}+\gamma G_{t+1}
$$

is the seed from which the Bellman equations and most of the book's algorithms grow.

## 2. Evaluating a policy with values

Before improving a policy, ask: **How good is it?**

The **state-value function** measures the expected return when the agent starts in state $s$ and then follows $\pi$:

$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s].
$$

The **action-value function** also fixes the first action:

$$
q_\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

Their relationship is intuitive:

$$
v_\pi(s)=\sum_a \pi(a\mid s)q_\pi(s,a).
$$

The state value is simply the policy-weighted average of the available action values.

### Bellman expectation equation

Substitute $G_t=R_{t+1}+\gamma G_{t+1}$ into the value definition:

$$
v_\pi(s)
=\sum_a\pi(a\mid s)
\sum_{s'}p(s'\mid s,a)
\left[r(s,a,s')+\gamma v_\pi(s')\right].
$$

In words:

> value now = expected immediate reward + discounted expected value later.

This is a system of self-consistency equations: every state's value depends on the values of possible successor states. For a finite MDP it can be written as

$$
\mathbf v_\pi=\mathbf r_\pi+\gamma P_\pi\mathbf v_\pi,
$$

and therefore, in principle,

$$
\mathbf v_\pi=(I-\gamma P_\pi)^{-1}\mathbf r_\pi.
$$

The direct inverse explains the mathematical solution, but iterative and sample-based methods are usually more practical.

## 3. Optimal values and optimal policies

Policy evaluation asks how good a given policy is. **Control** asks for the best policy.

Define the optimal values by

$$
v_*(s)=\max_\pi v_\pi(s),
\qquad
q_*(s,a)=\max_\pi q_\pi(s,a).
$$

The Bellman optimality equation replaces the policy's average over actions with a maximum:

$$
v_*(s)=\max_a\sum_{s'}p(s'\mid s,a)
\left[r(s,a,s')+\gamma v_*(s')\right].
$$

Similarly,

$$
q_*(s,a)=\sum_{s'}p(s'\mid s,a)
\left[r(s,a,s')+\gamma\max_{a'}q_*(s',a')\right].
$$

Once $q_*$ is known, acting optimally is easy:

$$
\pi_*(s)\in\arg\max_a q_*(s,a).
$$

The difficult part is estimating the values accurately enough to make this greedy choice.

> [!important] Expectation versus optimality
> A Bellman **expectation** equation evaluates the actions selected by a particular policy. A Bellman **optimality** equation selects the best action. This one change—from an average to a maximum—marks the move from prediction to control.

## 4. Planning when the model is known

If $p$ and $r$ are known, the agent can update values without interacting with the real environment. This is **dynamic programming**.

### Value iteration

Repeatedly apply the Bellman optimality backup:

$$
v_{k+1}(s)=\max_a\sum_{s'}p(s'\mid s,a)
\left[r(s,a,s')+\gamma v_k(s')\right].
$$

Each sweep pushes the value estimate toward $v_*$. After convergence, extract a greedy policy.

### Policy iteration

Alternate between two steps:

1. **Policy evaluation:** compute or approximate $v_\pi$ for the current policy.
2. **Policy improvement:** choose actions greedily with respect to that value.

The policy improvement theorem guarantees that the greedy policy is no worse than the old one. Repeating evaluation and improvement eventually produces an optimal policy in a finite discounted MDP.

## 5. Learning from complete episodes: Monte Carlo

When the model is unknown, the expectation in a Bellman equation cannot be calculated directly. One solution is to run the policy, observe full episodes, and use the actual return $G_t$ as a sample of value.

For a visited state:

$$
V(S_t)\leftarrow V(S_t)+\alpha\left[G_t-V(S_t)\right].
$$

For control, estimate action values in the same way:

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\left[G_t-Q(S_t,A_t)\right].
$$

Monte Carlo (MC) learning is conceptually simple and its target is based on real rewards, but it must wait until an episode ends. Its estimates can also have high variance because many random events affect a complete return.

### Exploration is necessary

Always choosing the current best-looking action can prevent the agent from discovering better alternatives. A common compromise is an **$\epsilon$-greedy policy**:

- with probability $1-\epsilon$, select a greedy action;
- with probability $\epsilon$, explore, usually by sampling an action uniformly.

The deeper tension is:

> **Exploitation** uses what the agent currently knows; **exploration** gathers information that may improve future decisions.

## 6. Stochastic approximation: the common engine

Many RL updates have the same shape:

$$
\text{new estimate}
=\text{old estimate}
+\text{step size}\times(\text{target}-\text{old estimate}).
$$

The term in parentheses is an **error** or **innovation**. Stochastic approximation explains why small, noisy corrections can converge to the desired solution.

For classical convergence in a stationary setting, a typical step-size requirement is

$$
\sum_{t=0}^{\infty}\alpha_t=\infty,
\qquad
\sum_{t=0}^{\infty}\alpha_t^2<\infty.
$$

The first condition prevents learning from stopping too soon; the second prevents noise from dominating forever. A small constant step size may not converge exactly, but it can adapt better when the environment changes.

Stochastic gradient descent (SGD) follows the same principle, replacing an exact gradient with a sample-based estimate.

## 7. Learning one step at a time: temporal difference

Temporal-difference (TD) learning combines two ideas:

- like MC, it learns from sampled experience without a model;
- like dynamic programming, it bootstraps from an existing value estimate.

The one-step TD error is

$$
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t),
$$

and TD(0) updates

$$
V(S_t)\leftarrow V(S_t)+\alpha\delta_t.
$$

Unlike MC, TD can learn before an episode finishes. The price is that its target contains an estimate, so errors can temporarily reinforce other errors.

### The major TD control algorithms

| Algorithm      | One-step target                                      | Learns the value of                  | Type       |
| -------------- | ---------------------------------------------------- | ------------------------------------ | ---------- |
| Sarsa          | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$                  | The policy actually taking $A_{t+1}$ | On-policy  |
| Expected Sarsa | $R_{t+1}+\gamma\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a)$ | The current policy in expectation    | On-policy  |
| Q-learning     | $R_{t+1}+\gamma\max_aQ(S_{t+1},a)$                   | A greedy target policy               | Off-policy |

All three use

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)
+\alpha\left[\text{target}-Q(S_t,A_t)\right].
$$

The labels **on-policy** and **off-policy** distinguish two policies:

- the **behavior policy** generates experience;
- the **target policy** is the policy being evaluated or improved.

Sarsa learns about the same exploratory policy that generates its data. Q-learning may behave exploratorily while learning about the greedy policy.

Multi-step methods sit between one-step TD and full-return MC. A longer target uses more observed rewards and less bootstrapping, trading lower bias for higher variance.

## 8. Scaling with value-function approximation

A table needs one entry per state or state–action pair. That becomes impossible for large or continuous spaces. Instead, represent values with parameters:

$$
\hat v(s;\mathbf w)
\quad\text{or}\quad
\hat q(s,a;\mathbf w).
$$

The same parameters are shared across many states, allowing the model to **generalize** from visited states to similar unseen ones. For a target $Y_t$, a semi-gradient update is

$$
\mathbf w_{t+1}
=\mathbf w_t
+\alpha\left[Y_t-\hat v(S_t;\mathbf w_t)\right]
\nabla_{\mathbf w}\hat v(S_t;\mathbf w_t).
$$

The approximator can be linear or nonlinear. A neural network that approximates action values gives the basic idea behind a deep Q-network (DQN).

DQN adds two important stabilizers:

- **experience replay:** store transitions and train on randomly sampled past experience, reducing temporal correlation and reusing data;
- **target network:** hold a separate, slowly updated network fixed while constructing TD targets, preventing the target from moving at every gradient step.

## 9. Optimizing the policy directly

Value-based methods learn values and then obtain a policy by acting greedily. A **policy-gradient** method instead defines a differentiable policy $\pi_\theta(a\mid s)$ and adjusts its parameters to increase expected performance $J(\theta)$.

The policy-gradient theorem leads to the central form

$$
\nabla_\theta J(\theta)
\propto
\mathbb E_{\pi_\theta}
\left[
Q^{\pi_\theta}(S,A)
\nabla_\theta\log\pi_\theta(A\mid S)
\right].
$$

Interpretation:

- if an action produces a better-than-usual result, increase its probability;
- if it produces a worse result, decrease its probability.

REINFORCE replaces the unknown action value with a sampled return:

$$
\theta\leftarrow\theta
+\alpha G_t\nabla_\theta\log\pi_\theta(A_t\mid S_t).
$$

This estimate is unbiased under standard assumptions, but often noisy. Subtracting a state-dependent **baseline** does not change the expected gradient and can reduce variance. A common choice is $v_\pi(s)$, producing the advantage

$$
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s).
$$

The advantage asks a more useful question than raw return: _Was this action better or worse than what is normally expected in this state?_

## 10. Actor–critic: putting both views together

Actor-critic methods combine:

- an **actor**, which represents and improves the policy; and
- a **critic**, which estimates values and judges the actor's choices.

A simple discounted actor-critic method can use the TD error as an estimate of advantage:

$$
\delta_t=R_{t+1}+\gamma\hat v(S_{t+1};\mathbf w)-\hat v(S_t;\mathbf w).
$$

Then update both parts:

$$
\begin{aligned}
\mathbf w &\leftarrow \mathbf w
+\beta\delta_t\nabla_{\mathbf w}\hat v(S_t;\mathbf w),\\
\theta &\leftarrow \theta
+\alpha\delta_t\nabla_\theta\log\pi_\theta(A_t\mid S_t).
\end{aligned}
$$

The critic learns how good states are; the actor makes actions judged positively by the critic more likely. This pairing usually learns more frequently and with lower variance than waiting for a complete REINFORCE return.

The chapter extends this idea through advantage actor-critic, off-policy correction with importance sampling, and deterministic actor-critic methods.

## My main takeaways

1. **The Bellman equation is the organizing principle.** It turns a long-term objective into a local relationship between the present and the future.
2. **Most algorithms differ mainly in how they construct a target.** The update pattern itself changes very little.
3. **Policy evaluation and policy improvement form a reusable loop.** This loop appears in dynamic programming, MC control, TD control, and actor-critic.
4. **Learning methods trade bias, variance, data efficiency, and stability.** There is no universally best target or representation.
5. **Actor-critic is a natural endpoint of the progression.** It combines value learning's feedback signal with direct policy optimization.

## Glossary

| Term           | Plain-language meaning                                               |
| -------------- | -------------------------------------------------------------------- |
| Agent          | The learner and decision-maker                                       |
| Environment    | Everything the agent interacts with                                  |
| State          | Information used to describe the current situation                   |
| Action         | A choice available to the agent                                      |
| Reward         | Immediate feedback after an action                                   |
| Return         | Accumulated discounted future reward                                 |
| Policy         | A rule or distribution for choosing actions                          |
| Value function | Expected return from a state or state–action pair                    |
| Model          | Transition and reward dynamics of the environment                    |
| Bellman backup | An update using immediate reward and successor value                 |
| Bootstrapping  | Updating an estimate using another current estimate                  |
| On-policy      | Learning about the policy that generates the experience              |
| Off-policy     | Learning about a policy different from the one generating experience |

## Source and further study

- Shiyu Zhao, [_Mathematical Foundations of Reinforcement Learning_](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) — official repository with the complete book, individual chapters, lecture slides, videos, errata, and grid-world code.
- [Grid-world code](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/tree/main/Code%20for%20grid%20world) — a compact environment for implementing and comparing the algorithms.
