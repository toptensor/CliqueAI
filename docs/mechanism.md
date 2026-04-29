
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
    -   Time: Time constraints. Currently, the time limit is sampled from ${6s, 7.5s, 10s, 15s, 30s}$.

## Difficulty Level

### Rationale

We evaluate difficulty level, $d(p)$, of each problem on a scale from 0 to greater than 1 (i.e., unbounded above) using predefined rules. The rules consider graph scale, existing solutions (available on some special graphs), time limits, and current miner performance.

We continuously refine this formula based on miner performance and internal experiments to make it more accurate.

### Latest

Currently, we have 3 types of problems:

-   Medium (difficulty=0.2): general graph with $290≤|V|≤300$
-   Hard (difficulty=0.4): general graph with $490≤|V|≤500$
-   Very Hard (difficulty=1): general graph with $690≤|V|≤700$

## Selection Process

The validator determines the problem type, then selects one problem of that type.

### Type Selection

For a set of problem types \(T\), each type is selected with equal probability:
$${A(t)=\frac{1}{|T|},\quad \forall\, t\in T}$$

### Problem Selection

The validator retrieves a random problem of type $t$ from our database. **Each problem is retrieved at most once**, meaning each problem appears at most once even across different validators.

# Miner Selection

To lower the entry barrier for new participants, we sample eligible miners with equal probability for each problem. The probability is adjusted by problem difficulty.

Before sampling, the validator filters out validator nodes and recently updated miners. The remaining miners are eligible to receive the problem.

In brief, miner selection includes the following steps:

1.  Filter out validator nodes and miners updated within the current epoch.
2.  Calculate a difficulty-adjusted selection probability.
3.  Independently sample each eligible miner with that same probability.

## Miner Sampling

The probability that an eligible miner receives a problem $p$, denoted as $P(p)$, depends only on the problem's difficulty level, $d(p)$, and a fixed reference parameter, $r_{ref}$.

We use:

$$ r_{ref} = 1.5 $$

$$ x = \sqrt{1 + r_{ref}} $$

The selection probability is:

$$ P(p) = 1 - e^{-max(0, x-d(p)-0.5)} $$

Every eligible miner receives the same probability $P(p)$ for a given problem. Harder problems produce lower selection probabilities, while easier problems are distributed to more miners on average.

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

With these principles in mind, we aggregate $\omega$ and $\delta$ using:

$$ f(m, p) = \omega_n(m)(1+d(p))+\delta_n(m) $$

where $f(m, p)$ represents miner $m$'s final score on problem $p$.

# Weight Setting

We calculate a miner's rating using the Debiased EMA of their historical solution scores.

$$ y_t = \alpha f(m, p) + (1 - \alpha)y_{t-1} $$

$$ \hat{y}_t = \frac{y_t}{1 - (1 - \alpha)^t} $$

with $\alpha=0.01$, which means half-life of each sample is about 69 samples.
