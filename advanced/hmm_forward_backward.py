r"""
A. **Mission** (Step 0: Understand the Problem)
While the Viterbi algorithm identifies the single most likely hidden path through an HMM trellis, the **Forward-Backward Algorithm** solves two fundamentally deeper probabilistic challenges:
1. **The Evaluation Problem**: Computing the exact total marginal probability of an observation sequence $P(\mathbf{O} \mid \lambda) = \sum_{Q} P(\mathbf{O}, Q \mid \lambda)$ summing across all possible hidden paths.
2. **The Posterior Decoding Problem**: Computing the posterior state responsibility $\gamma_t(i) = P(q_t = i \mid \mathbf{O}, \lambda)$—the probability that the model was in state $i$ at time $t$ conditioned on the *entire* observation history (past, present, and future).
Crucially, these posteriors form the E-step of the **Baum-Welch Algorithm** (an instance of Expectation-Maximization), which allows an HMM to learn its transition matrix $\mathbf{A}$, emission matrix $\mathbf{B}$, and initial probabilities $\boldsymbol{\pi}$ in a completely unsupervised manner from raw observation sequences.
Your mission is to implement `HMMForwardBackward` from scratch using only NumPy. You will derive the forward and backward recurrences, implement dynamic scaling factors to eliminate floating-point underflow, accumulate expected counts across multiple sequences, and train the model using Baum-Welch EM.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — 2D and 3D tensor indexing, row-wise normalization, and clipping.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Matrix-vector products ($\boldsymbol{\alpha}_t^T \mathbf{A}$) and outer products.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Constrained optimization with Lagrange multipliers (ensuring rows of stochastic matrices sum to 1).
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) — Marginalization, conditional expectation, and latent variable models.
- [Chapter 7: Viterbi](./hmm_viterbi.py) — Understanding HMM trellis structure and Dataset H1.

C. **Learning Questions** (Step 1: Build Intuition)
1. **Past vs. Future Conditioning**: What observations does the forward variable $\alpha_t(i)$ condition on versus the backward variable $\beta_t(i)$? Why does their product $\alpha_t(i)\beta_t(i)$ capture information from the entire timeline $[1, T]$?
2. **Why Underflow Occurs**: In a sequence of length $T=100$, why does raw $\alpha_{100}(i)$ underflow to zero in 64-bit floating point arithmetic?
3. **The Scaling Trick (Rabiner, 1989)**: How does normalizing the forward vector at each time step $t$ by scale factor $s_t = \sum_i \alpha_t(i)$ keep all values well within dynamic float range while allowing sequence log-likelihood to be recovered via $\ln P(\mathbf{O}) = \sum_{t=1}^T \ln s_t$?
4. **Baum-Welch as EM**: What are the "hidden data" in Baum-Welch? Why does taking expected counts of state transitions and emissions guarantee monotonic improvement in sequence log-likelihood?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand (Dataset H1)**:
   Consider Dataset H1 parameters:
   States: $S = \{\text{Sunny} (0), \text{Rainy} (1)\}$.
   Observations: $V = \{\text{Walk} (0), \text{Shop} (1), \text{Clean} (2)\}$.
   $\boldsymbol{\pi} = [0.6, 0.4]$, $\mathbf{A} = \begin{bmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{bmatrix}$, $\mathbf{B} = \begin{bmatrix} 0.6 & 0.3 & 0.1 \\ 0.1 & 0.4 & 0.5 \end{bmatrix}$.
   Observation: $\mathbf{O} = [\text{Walk} (0), \text{Shop} (1), \text{Clean} (2)]$.
   - **Forward Pass with Scaling ($t=1$, $o_1 = 0$)**:
     $\alpha_1(0) = \pi_0 B_{0, 0} = 0.6 \times 0.6 = 0.36$.
     $\alpha_1(1) = \pi_1 B_{1, 0} = 0.4 \times 0.1 = 0.04$.
     Scale factor $s_1 = \alpha_1(0) + \alpha_1(1) = 0.36 + 0.04 = 0.40$.
     Scaled $\hat{\alpha}_1 = [0.36/0.40, 0.04/0.40] = [0.90, 0.10]$.
   - **Forward Pass ($t=2$, $o_2 = 1$)**:
     $\alpha_2(0) = (\hat{\alpha}_1(0) A_{0, 0} + \hat{\alpha}_1(1) A_{1, 0}) B_{0, 1} = (0.9(0.7) + 0.1(0.4)) \times 0.3 = (0.63 + 0.04) \times 0.3 = 0.67 \times 0.3 = 0.201$.
     $\alpha_2(1) = (\hat{\alpha}_1(0) A_{0, 1} + \hat{\alpha}_1(1) A_{1, 1}) B_{1, 1} = (0.9(0.3) + 0.1(0.6)) \times 0.4 = (0.27 + 0.06) \times 0.4 = 0.33 \times 0.4 = 0.132$.
     Scale factor $s_2 = 0.201 + 0.132 = 0.333$.
     Scaled $\hat{\alpha}_2 = [0.201/0.333, 0.132/0.333] \approx [0.6036, 0.3964]$.
   - **Forward Pass ($t=3$, $o_3 = 2$)**:
     $\alpha_3(0) = (0.6036(0.7) + 0.3964(0.4)) \times 0.1 = (0.4225 + 0.1586) \times 0.1 = 0.5811 \times 0.1 = 0.05811$.
     $\alpha_3(1) = (0.6036(0.3) + 0.3964(0.6)) \times 0.5 = (0.1811 + 0.2378) \times 0.5 = 0.4189 \times 0.5 = 0.20945$.
     Scale factor $s_3 = 0.05811 + 0.20945 = 0.26756$.
     Scaled $\hat{\alpha}_3 = [0.05811/0.26756, 0.20945/0.26756] \approx [0.2172, 0.7828]$.
   - **Sequence Probability & Log-Likelihood**:
     $$P(\mathbf{O} \mid \lambda) = s_1 \times s_2 \times s_3 = 0.40 \times 0.333 \times 0.26756 \approx 0.035639$$
     $$\ln P(\mathbf{O} \mid \lambda) = \ln(0.40) + \ln(0.333) + \ln(0.26756) \approx -0.91629 - 1.09961 - 1.31842 = -3.33432$$
   - **Backward Pass Initialization ($t=3$)**:
     $\hat{\beta}_3 = [1.0, 1.0] / s_3 = [1.0/0.26756, 1.0/0.26756] \approx [3.7375, 3.7375]$.
     Notice that the posterior probability of states at $T=3$ is:
     $\gamma_3(0) = \hat{\alpha}_3(0) \hat{\beta}_3(0) s_3 = 0.2172 \times 1.0 = 0.2172$.
     $\gamma_3(1) = \hat{\alpha}_3(1) \hat{\beta}_3(1) s_3 = 0.7828 \times 1.0 = 0.7828$.
     Conditioned on seeing $[\text{Walk}, \text{Shop}, \text{Clean}]$, the probability that Day 3 was Rainy is $\approx 78.3\%$!

2. **Step 3: Mathematical Notation**:
   - States: $\mathcal{S} = \{0, 1, \dots, N_s - 1\}$.
   - Observations: $\mathcal{V} = \{0, 1, \dots, N_v - 1\}$.
   - Sequence: $\mathbf{O} = [o_1, \dots, o_T]$.
   - Parameters: $\lambda = (\mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$.
   - Forward variables: $\alpha_t(i) = P(o_1, \dots, o_t, q_t = i \mid \lambda)$.
   - Backward variables: $\beta_t(i) = P(o_{t+1}, \dots, o_T \mid q_t = i, \lambda)$.
   - State posterior: $\gamma_t(i) = P(q_t = i \mid \mathbf{O}, \lambda)$.
   - Transition posterior: $\xi_t(i, j) = P(q_t = i, q_{t+1} = j \mid \mathbf{O}, \lambda)$.

3. **Step 4: Deriving the Scaled Forward-Backward & Baum-Welch M-Step**:
   - **Scaled Forward Recurrence**:
     For $t=1$: $\alpha_1(i) = \pi_i B_{i, o_1}$; $s_1 = \sum_i \alpha_1(i)$; $\hat{\alpha}_1(i) = \alpha_1(i) / s_1$.
     For $t=2 \dots T$:
     $$\alpha_t(j) = \left( \sum_{i=1}^{N_s} \hat{\alpha}_{t-1}(i) A_{ij} \right) B_{j, o_t}$$
     $$s_t = \sum_{j=1}^{N_s} \alpha_t(j), \quad \hat{\alpha}_t(j) = \frac{\alpha_t(j)}{s_t}$$
     $$\ln P(\mathbf{O} \mid \lambda) = \sum_{t=1}^T \ln s_t$$
   - **Scaled Backward Recurrence**:
     For $t=T$: $\hat{\beta}_T(i) = 1.0$ (or scaled by $s_T$).
     For $t = T-1$ down to $1$:
     $$\beta_t(i) = \sum_{j=1}^{N_s} A_{ij} B_{j, o_{t+1}} \hat{\beta}_{t+1}(j)$$
     $$\hat{\beta}_t(i) = \frac{\beta_t(i)}{s_{t+1}}$$
   - **Posterior Responsibilities**:
     $$\gamma_t(i) = \frac{\hat{\alpha}_t(i) \hat{\beta}_t(i)}{\sum_{j=1}^{N_s} \hat{\alpha}_t(j) \hat{\beta}_t(j)} = \hat{\alpha}_t(i) \hat{\beta}_t(i)$$
     $$\xi_t(i, j) = \frac{\hat{\alpha}_t(i) A_{ij} B_{j, o_{t+1}} \hat{\beta}_{t+1}(j)}{\sum_{k=1}^{N_s} \sum_{l=1}^{N_s} \hat{\alpha}_t(k) A_{kl} B_{l, o_{t+1}} \hat{\beta}_{t+1}(l)}$$
   - **Baum-Welch M-Step Updates (Accumulated across sequences $m=1 \dots M$)**:
     $$\pi_i^* = \frac{\sum_{m} \gamma_{1}^{(m)}(i)}{M}$$
     $$A_{ij}^* = \frac{\sum_{m} \sum_{t=1}^{T_m-1} \xi_t^{(m)}(i, j)}{\sum_{m} \sum_{t=1}^{T_m-1} \gamma_t^{(m)}(i)}$$
     $$B_{j k}^* = \frac{\sum_{m} \sum_{t=1, o_t^{(m)}=k}^{T_m} \gamma_t^{(m)}(j)}{\sum_{m} \sum_{t=1}^{T_m} \gamma_t^{(m)}(j)}$$

E. **Implementation Contract**
- Class: `HMMForwardBackward`
- Constructor:
  - `__init__(self, n_states: int, n_observations: int, random_state: int = 42)`:
    Initializes model parameters randomly (with positive noise) ensuring each row sums to 1.0.
- Attributes:
  - `initial_`: ndarray of shape `(n_states,)`, prior probabilities.
  - `transition_`: ndarray of shape `(n_states, n_states)`, state transition probabilities.
  - `emission_`: ndarray of shape `(n_states, n_observations)`, emission probabilities.
  - `log_likelihood_history_`: list of floats recording total log-likelihood per Baum-Welch iteration.
- Methods:
  - `forward(self, observations: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray, float]`:
    Computes scaled forward trellis $\hat{\boldsymbol{\alpha}}$ of shape `(T, N_s)`, scale factors $\mathbf{s}$ of shape `(T,)`, and sequence log-likelihood.
  - `backward(self, observations: list[int] | np.ndarray, scales: np.ndarray) -> np.ndarray`:
    Computes scaled backward trellis $\hat{\boldsymbol{\beta}}$ of shape `(T, N_s)` using scale factors from forward pass.
  - `compute_posteriors(self, alpha_scaled: np.ndarray, beta_scaled: np.ndarray, observations: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]`:
    Computes state responsibilities $\boldsymbol{\gamma}$ of shape `(T, N_s)` and transition responsibilities $\boldsymbol{\xi}$ of shape `(T-1, N_s, N_s)`.
  - `fit(self, sequences: list[list[int] | np.ndarray], max_iters: int = 50, tol: float = 1e-4) -> self`:
    Executes Baum-Welch EM algorithm across list of sequences until log-likelihood change $< tol$ or `max_iters` reached.

F. **Guided Implementation Stages**
**Step 5: Algorithm Design & Pseudocode**
```text
fit(sequences):
  For iter in range(max_iters):
    total_log_likelihood = 0.0
    accum_pi = zeros(N_s)
    accum_A_num = zeros(N_s, N_s)
    accum_A_den = zeros(N_s)
    accum_B_num = zeros(N_s, N_v)
    accum_B_den = zeros(N_s)
    
    # E-step over all sequences:
    For seq in sequences:
      alpha_hat, scales, ll = forward(seq)
      beta_hat = backward(seq, scales)
      gamma, xi = compute_posteriors(alpha_hat, beta_hat, seq)
      
      total_log_likelihood += ll
      accum_pi += gamma[0]
      accum_A_num += sum(xi, axis=0)
      accum_A_den += sum(gamma[:-1], axis=0)
      For k in range(N_v):
        mask = (seq == k)
        accum_B_num[:, k] += sum(gamma[mask], axis=0)
      accum_B_den += sum(gamma, axis=0)
      
    record total_log_likelihood
    if |current_ll - prev_ll| < tol:
      break
      
    # M-step update:
    initial_ = accum_pi / len(sequences)
    transition_ = accum_A_num / (accum_A_den[:, None] + 1e-12)
    emission_ = accum_B_num / (accum_B_den[:, None] + 1e-12)
```

**Checkpoint 1: Scaled Forward Pass and Log-Likelihood**
- **What to learn**: Implementing normalized forward variables and recovering sequence log-likelihood.
- **What to do**: Implement `forward` using normalization $s_t = \sum_i \alpha_t(i)$ at each time step.
- **How to check yourself**: Run on Dataset H1 with parameters from hand derivation. Verify scale factors match $s = [0.40, 0.333, 0.26756]$ and log-likelihood matches $\approx -3.33432$.
- **When to proceed**: Forward pass matches hand calculations exactly.
- **Recovery hints**:
  - *Hint 1 (Concept)*: `s_t` is the sum of unnormalized forward values at time $t$.
  - *Hint 2 (Operation)*: `ll = np.sum(np.log(scales))` (or `-sum(np.log(c_t))` if scale is $1/\text{sum}$).
  - *Hint 3 (Debugging)*: Check `alpha_scaled[t]` sums to 1.0 at every time step $t$.

**Checkpoint 2: Scaled Backward Pass and Posteriors**
- **What to learn**: Backward dynamic programming and calculating $\gamma$ and $\xi$.
- **What to do**: Implement `backward` and `compute_posteriors`.
- **How to check yourself**: Verify that $\sum_j \xi_t(i, j) == \gamma_t(i)$ for all $t < T$, and $\sum_i \gamma_t(i) == 1.0$ for all $t$.
- **When to proceed**: Posteriors satisfy mathematical consistency identities.
- **Recovery hints**:
  - *Hint 1 (Concept)*: The sum of transitions leaving state $i$ must equal the probability of being in state $i$.
  - *Hint 2 (Operation)*: Use `np.allclose(np.sum(xi, axis=2), gamma[:-1])`.
  - *Hint 3 (Debugging)*: In `backward`, scale using `scales[t+1]` (or corresponding forward scale).

**Checkpoint 3: Baum-Welch EM Convergence**
- **What to learn**: Accumulating expected counts across multiple sequences and validating EM monotonicity.
- **What to do**: Implement the M-step updates in `fit()` and run on 10 synthetic sequences generated from an HMM.
- **How to check yourself**: Verify that `log_likelihood_history_` never decreases across iterations.
- **When to proceed**: Log-likelihood is strictly non-decreasing and parameters converge.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Baum-Welch is guaranteed to increase or maintain sequence log-likelihood.
  - *Hint 2 (Operation)*: Add tiny epsilon (`1e-12`) to denominators to prevent division by zero for unvisited states.
  - *Hint 3 (Debugging)*: Re-normalize rows of `transition_` and `emission_` so each sums to 1.0.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_forward_likelihood_exact`: Verify `forward([0, 1, 2])` on Dataset H1 parameters yields log-likelihood $\approx -3.33432$.
2. `test_posterior_identities`: For any sequence, verify $\sum_i \gamma_t(i) = 1.0$ and $\sum_j \xi_t(i, j) = \gamma_t(i)$.
3. `test_baum_welch_monotonicity`: Verify `log_likelihood_history_` is monotonically non-decreasing over 20 Baum-Welch iterations.
4. `test_variable_length_sequences`: Pass sequences of lengths $[10, 50, 100]$ to `fit()` and confirm graceful execution.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Forward-Backward per sequence of length $T$: $O(T \cdot N_s^2)$ operations.
  - E-step for $M$ sequences of average length $T$: $O(M \cdot T \cdot N_s^2)$.
  - M-step: $O(N_s^2 + N_s \cdot N_v)$ operations.
  - Total per EM iteration: $O(M \cdot T \cdot N_s^2 + N_s \cdot N_v)$.
  - Total for $I$ iterations: $O(I \cdot M \cdot T \cdot N_s^2)$.
- **Space Complexity**:
  - Forward and backward trellises per sequence: $O(T \cdot N_s)$.
  - Transition posteriors $\boldsymbol{\xi}$: $O(T \cdot N_s^2)$.
  - Model parameters: $O(N_s^2 + N_s \cdot N_v + N_s)$.
- **Trade-offs**: Efficiently learns latent dynamics without state supervision, but EM can get trapped in local likelihood maxima depending on initialization.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement scaled forward pass and verify log-likelihood on Dataset H1.
- **Level 2 (Standard)**: Implement backward pass, posteriors ($\gamma, \xi$), and full Baum-Welch `fit()` supporting multiple sequences.
- **Level 3 (Challenge)**: Generate synthetic observation sequences using ancestral sampling from a known ground-truth HMM, fit Baum-Welch, and demonstrate parameter recovery (up to state permutation).
- **Definition of Done**: Forward log-likelihood matches Dataset H1, Baum-Welch monotonically improves log-likelihood, and handles variable-length sequences without underflow.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why is Baum-Welch susceptible to local maxima, and what strategies (e.g. multiple random restarts, Dirichlet priors) mitigate this issue?
2. If observations are continuous rather than discrete, how does the emission update change from counting occurrences to estimating Gaussian means and variances? (Hint: Continuous HMM / GMM-HMM).
"""
