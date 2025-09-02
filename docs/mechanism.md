
# Overview

A problem distribution includes 4 steps:

1.  **Problem Selection**
    The validator selects a problem from the database.

2.  **Miner Selection**
    The validator selects miners to receive the problem.

3.  **Scoring**
    The validator evaluates miner solutions.

4.  **Weight Setting**
    The validator modifies miner weights.


We will walk through their details in following sections.

# Problem Selection

## Problem Attributes

Problems are classified according to the following attributes:

-   **Scale**
    -   $|V|$: Number of vertices.
    -   $|E|$: Number of edges.
-   **Label**
    -   General: general graph without guaranteed features.
-   **Limit**
    -   Time: Time constraints

## Difficulty Level

### Rationale

We evaluate difficulty level, $d(p)$, of each problem on a scale from 0 to 1 using predefined rules. The rules consider graph scale, existing solutions (available on some special graphs), time limits, and current miner performance.

We continuously refine this formula based on miner performance and internal experiments to make it more accurate.

### Latest (since 2025/08/25)

Currently, we have only 3 types of problems:

-   Easy (difficulty=0.1): general graph with $90≤|V|≤100$
-   Medium (difficulty=0.2): general graph with $290≤|V|≤300$
-   Hard (difficulty=0.4): general graph with $490≤|V|≤500$

## Selection Process

The validator determines the problem type, then selects one problem of that type.

### Type Selection

The validator tends to assign difficult problems to experienced miners (explained in the next section). This approach creates these outcomes:

-   Harder problems are distributed to fewer miners.
-   New miners receive many easy problems but very few hard problems.

However, this can lead to a pooling equilibrium. Outstanding new miners may find it extremely difficult to prove themselves because they receive ZERO hard problems to distinguish themselves from other new miners.

To address this issue, we boost the appearance of hard problems so new miners have higher chances to receive harder problems to showcase their abilities.

A problem with type $t$ has appearance probability, $A(t)$, inversely proportional to its expected number of selected miners. This is determined by its difficulty level, $d(t)$, and current miner experience levels, $x(m)$ (explained in the next section).

$$ A'(t)=\frac{1}{\sum\limits_{m\in M}{1 - e^{-max(0, x(m)-d(t)-0.5)}}} $$

$$ A(t) = \frac{A'(t)}{\sum\limits_{u\in T}{A'(u)}} $$

where $M$ is the set of all miners and $T$ is the set of all problem types.

### Problem Selection

The validator retrieves a random problem of type $t$ from our database. **Each problem is retrieved at most once**, meaning each problem appears at most once even across different validators.

# Miner Selection

To lower the entry barrier for new participants, we assign experienced miners problems with higher frequency and difficulty.

New miners can focus on basic problems during their initial participation and gradually receive more diverse and difficult problems as they demonstrate their computational ability.

Since miners with higher historical rewards naturally have higher stake, we use alpha stake as a proxy for miner experience.

In brief, miner selection includes the following steps:

1.  Calculate each miner's alpha stake.
2.  Sample miners to distribute the problem accordingly.

Let's examine each step in detail.

## Stake Calculation

Miners can earn more by staking their alphas on validators, while the official validator benefits from greater impact to stabilize scoring quality.

Our win-win proposal: we include a miner's stake on our validator when calculating their total miner stake.

Here's how we calculate a miner's alpha stake:

-   Suppose a miner registers with coldkey=`C` and hotkey=`H`.
-   The amount of alpha that `C` stakes on `H` is `S_miner`.
-   The amount of alpha that `C` stakes on our validator is `S_validator`.
-   The number of miners registered by `C` is `N_miner`.

The miner's alpha stake would be `S_miner` + `S_validator` / `N_miner`. This means the credit for alphas staked on our validator by a coldkey is divided equally among all its miners.

## Miner Sampling

The probability that a miner $m$ receives a problem $p$, denoted as $P(m, p)$, depends on the miner's experience level, $x(m)$, and the problem's difficulty level, $d(p)$.

We calculate $x(m)$ using the miner's alpha stake, $s_m$, and the average of all miners' alpha stakes, $\overline{S}$:

$$ x(m) = \sqrt{1 + \frac{s_m}{\overline{S}}} $$

The probability is then determined by:

$$ P(m, p) = 1 - e^{-max(0, x(m)-d(p)-0.5)} $$

With this formula, miners with more experience receive both a greater number and variety of problems.

-   Newbie miners (with zero stake) only receive problems where $d(p) < 0.5$.
-   Average miners (with stake roughly equal to the average) can receive most problems where $d(p) < 0.914$.
-   Experienced miners receive not only more difficult problems but also more problems overall.
-   If all miners have no stakes, then $x(m):=1 \forall m \in M$

# Scoring

To empower the whole subnet, we consider both performance and diversity.

## Notations

-   A `solution` is a set of vertices $G = \{v_1, …, v_N\}$, where $N$ is the size of the clique.
-   $sol_m$: the solution returned by a miner $m$.
-   $N(sol_m)$: size of the solution returned by $m$ .
-   $M$: the set of all miners selected to solve the problem.

## Optimality

Optimality score, $\omega$, is defined as

$$ \omega (m)= val(sol_m)e^{-\frac{pr(sol_m)}{rel(sol_m)}} $$

-   $val(sol_m)$: whether the solution is valid and is a maximal clique.
-   $rel(sol_m)$: $\frac{N(sol_m)}{\max{\{N(sol_i) | i \in M}\}}$
-   $pr(sol_m)$: $\frac{|\{i \in M | N(sol_i) > N(sol_m)\}| }{|M|}$

After deriving all miners' optimality scores, we normalize them to ensure proper scaling.

$$ \omega_n(m) = \frac{\omega(m)}{ \max_{i \in M}\omega(i)} $$

## Diversity

Diversity score, $\delta$ , is defined as

$$ \delta(m) = val(sol_m)unq(sol_m) $$

-   $unq(sol_m)$: 1 divided by number of miners whose solution is exactly identical to $sol_m$.

Similar to optimality, we normalize the diversity scores.

$$ \delta_n(m) = \frac{\delta(m)}{ \max_{i \in M}\delta(i) } $$

## Aggregation

To incentivize miners to develop general, robust, and diverse solvers, we expect the scoring measure to satisfy the following properties:

-   Unique solvers should receive more credit when performance is similar.
-   Performance should always carry more weight than uniqueness.
-   For difficult problems, performance should be emphasized more than for easier problems, as they typically offer more room for improvement.
-   Since difficult problems appear less frequently, they should offer greater potential rewards to winners to prevent miners from undervaluing them.

With these principles in mind, we aggregate $\omega$ and $\delta$ using:

$$ f(m, p) = \omega_n(m)(1+d(p))+\delta_n(m) $$

where $f(m, p)$ represents miner $m$'s final score on problem $p$.

# Weight Setting

We calculate a miner's rating using the Debiased EMA of their historical solution scores.

$$ y_t = \alpha f(m, p) + (1 - \alpha)y_{t-1} $$

$$ \hat{y}_t = \frac{y_t}{1 - (1 - \alpha)^t} $$

with $\alpha=0.01$, which means half-life of each sample is about 69 samples.
