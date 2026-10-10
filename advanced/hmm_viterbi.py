r"""
A. **Mission** (Step 0: Understand the Problem)
The Hidden Markov Model (HMM) is a probabilistic sequence model that captures systems where an unobservable (latent) Markovian state process generates an observable sequence of discrete emissions.
The central decoding challenge in HMMs is: **Given a sequence of observations $\mathbf{O} = [o_1, o_2, \dots, o_T]$ and known model parameters $\lambda = (\mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$, what is the single most likely sequence of hidden states $Q^* = [q_1^*, q_2^*, \dots, q_T^*]$ that generated them?**
Naïve approaches fail catastrophically:
1. *Greedy decoding* (choosing $\operatorname{argmax} P(q_t \mid o_t)$ at each step) can yield illegal sequences with zero transition probability ($A_{q_{t-1}, q_t} = 0$).
2. *Exhaustive search* evaluates all $|S|^T$ possible paths, which exponentially explodes ($2^{100} \approx 10^{30}$ paths).
The **Viterbi Algorithm** (Andrew Viterbi, 1967) uses dynamic programming over a trellis to find the globally optimal state sequence in $O(T \cdot |S|^2)$ polynomial time.
Your mission is to implement `HMMViterbi` from scratch using only NumPy. You will formulate the recurrence in log-space to prevent underflow on long sequences, maintain backpointer pointers, backtrack to reconstruct the optimal state trajectory on Dataset H1, and handle edge cases such as impossible transitions.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — 2D array indexing, vectorization, `np.argmax`, and handling `-np.inf`.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Transition matrices and row-stochasticity.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Dynamic programming and Bellman's principle of optimality.
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) — Joint probability chains, conditional independence, and the Markov property.
- [Chapter 7: Overview](./README.md) — HMM Problem Formulation and Dataset H1 definition.

C. **Learning Questions** (Step 1: Build Intuition)
1. **The Trellis Representation**: How does representing state transitions across time as a directed layered graph (trellis) allow us to reuse intermediate subproblem solutions?
2. **Bellman's Principle of Optimality**: Why is the most likely path to state $k$ at time $t$ guaranteed to contain the most likely path to some state $j$ at time $t-1$? Why can all other suboptimal paths arriving at $j$ be permanently discarded?
3. **Log-Space Transformation**: Why does computing joint path probabilities in regular float space ($P = \pi \cdot b_1 \cdot a_1 \dots$) underflow to zero when $T > 50$? How does operating in log-probability space ($\ln(P) = \ln \pi + \ln b_1 + \ln a_1 \dots$) turn multiplications into additions and preserve numerical stability indefinitely?
4. **Decoding vs. Posterior Marginalization**: Why might the Viterbi path (the single sequence that maximizes joint probability $P(Q, \mathbf{O})$) differ from selecting the state that maximizes individual marginal probability $P(q_t \mid \mathbf{O})$ at each time step?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand (Dataset H1)**:
   - Hidden States: $S = \{\text{Sunny} (0), \text{Rainy} (1)\}$.
   - Observations: $V = \{\text{Walk} (0), \text{Shop} (1), \text{Clean} (2)\}$.
   - Initial probabilities: $\boldsymbol{\pi} = [0.6, 0.4]$.
   - Transition matrix:
     $$\mathbf{A} = \begin{bmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{bmatrix}$$
   - Emission matrix:
     $$\mathbf{B} = \begin{bmatrix} 0.6 & 0.3 & 0.1 \\ 0.1 & 0.4 & 0.5 \end{bmatrix}$$
   - Observation sequence: $\mathbf{O} = [\text{Walk} (0), \text{Shop} (1), \text{Clean} (2)]$.
   - **Time $t=1$ ($o_1 = \text{Walk} (0)$)**:
     - $V_1(0) = \pi_0 \times B_{0, 0} = 0.6 \times 0.6 = 0.36$.
     - $V_1(1) = \pi_1 \times B_{1, 0} = 0.4 \times 0.1 = 0.04$.
     - Backpointers: $BP_1(0) = 0, BP_1(1) = 0$ (arbitrary / start).
   - **Time $t=2$ ($o_2 = \text{Shop} (1)$)**:
     - For state 0 (Sunny):
       - From 0: $V_1(0) \times A_{0, 0} \times B_{0, 1} = 0.36 \times 0.7 \times 0.3 = 0.0756$.
       - From 1: $V_1(1) \times A_{1, 0} \times B_{0, 1} = 0.04 \times 0.4 \times 0.3 = 0.0048$.
       - Maximum: $V_2(0) = 0.0756$; Backpointer: $BP_2(0) = 0$ (Sunny).
     - For state 1 (Rainy):
       - From 0: $V_1(0) \times A_{0, 1} \times B_{1, 1} = 0.36 \times 0.3 \times 0.4 = 0.0432$.
       - From 1: $V_1(1) \times A_{1, 1} \times B_{1, 1} = 0.04 \times 0.6 \times 0.4 = 0.0096$.
       - Maximum: $V_2(1) = 0.0432$; Backpointer: $BP_2(1) = 0$ (Sunny).
   - **Time $t=3$ ($o_3 = \text{Clean} (2)$)**:
     - For state 0 (Sunny):
       - From 0: $V_2(0) \times A_{0, 0} \times B_{0, 2} = 0.0756 \times 0.7 \times 0.1 = 0.005292$.
       - From 1: $V_2(1) \times A_{1, 0} \times B_{0, 2} = 0.0432 \times 0.4 \times 0.1 = 0.001728$.
       - Maximum: $V_3(0) = 0.005292$; Backpointer: $BP_3(0) = 0$ (Sunny).
     - For state 1 (Rainy):
       - From 0: $V_2(0) \times A_{0, 1} \times B_{1, 2} = 0.0756 \times 0.3 \times 0.5 = 0.011340$.
       - From 1: $V_2(1) \times A_{1, 1} \times B_{1, 2} = 0.0432 \times 0.6 \times 0.5 = 0.012960$.
       - Maximum: $V_3(1) = 0.012960$; Backpointer: $BP_3(1) = 1$ (Rainy).
   - **Termination & Backtracking**:
     - At $t=3$, compare final states: $\max(V_3(0), V_3(1)) = \max(0.005292, 0.012960) = 0.012960$.
     - Best final state: $q_3^* = 1$ (Rainy).
     - Backtrack $t=2$: $q_2^* = BP_3(1) = 1$ (Rainy).
     - Backtrack $t=1$: $q_1^* = BP_2(1) = 0$ (Sunny).
     - Optimal hidden state sequence: $Q^* = [0, 1, 1] \implies [\text{Sunny}, \text{Rainy}, \text{Rainy}]$.
     - Total path joint probability: $0.012960$ ($\ln P \approx -4.3459$).

2. **Step 3: Mathematical Notation**:
   - Hidden state space: $\mathcal{S} = \{0, 1, \dots, N_s - 1\}$.
   - Observation vocabulary: $\mathcal{V} = \{0, 1, \dots, N_v - 1\}$.
   - Observation sequence: $\mathbf{O} = [o_1, o_2, \dots, o_T]$.
   - Initial probability vector: $\boldsymbol{\pi} \in \mathbb{R}^{N_s}$, where $\pi_i = P(q_1 = i)$.
   - Transition probability matrix: $\mathbf{A} \in \mathbb{R}^{N_s \times N_s}$, where $A_{ij} = P(q_t = j \mid q_{t-1} = i)$.
   - Emission probability matrix: $\mathbf{B} \in \mathbb{R}^{N_s \times N_v}$, where $B_{jk} = P(o_t = k \mid q_t = j)$.
   - Trellis values: $v_t(j) = \max_{q_1, \dots, q_{t-1}} \ln P(q_1, \dots, q_{t-1}, q_t = j, o_1, \dots, o_t \mid \lambda)$.
   - Backpointers: $\text{bp}_t(j) = \operatorname{argmax}_i [v_{t-1}(i) + \ln A_{ij}]$.

3. **Step 4: The Log-Space Viterbi Recurrence Relations**:
   - Log transformation: $\ln \pi_i = \ln(\pi_i)$, $\ln A_{ij} = \ln(A_{ij})$, $\ln B_{jk} = \ln(B_{jk})$ (with $\ln(0) = -\infty$).
   - **Initialization ($t=1$)**:
     $$v_1(j) = \ln \pi_j + \ln B_{j, o_1}, \quad \forall j \in \mathcal{S}$$
   - **Recursion ($t = 2, 3, \dots, T$)**:
     $$v_t(j) = \max_{i \in \mathcal{S}} \left[ v_{t-1}(i) + \ln A_{ij} \right] + \ln B_{j, o_t}, \quad \forall j \in \mathcal{S}$$
     $$\text{bp}_t(j) = \operatorname{argmax}_{i \in \mathcal{S}} \left[ v_{t-1}(i) + \ln A_{ij} \right]$$
   - **Termination**:
     $$\ln P^* = \max_{j \in \mathcal{S}} v_T(j)$$
     $$q_T^* = \operatorname{argmax}_{j \in \mathcal{S}} v_T(j)$$
   - **Backtracking ($t = T-1, T-2, \dots, 1$)**:
     $$q_t^* = \text{bp}_{t+1}(q_{t+1}^*)$$

E. **Implementation Contract**
- Class: `HMMViterbi`
- Constructor:
  - `__init__(self, transition_probs: np.ndarray, emission_probs: np.ndarray, initial_probs: np.ndarray)`:
    Converts inputs to NumPy arrays of floats.
    Computes log-space representations `log_A`, `log_B`, `log_pi` by safely replacing zero probabilities with `-np.inf`.
    Validates row stochasticity (rows sum to 1.0 within tolerance).
- Attributes:
  - `A`: ndarray of shape `(N_s, N_s)`.
  - `B`: ndarray of shape `(N_s, N_v)`.
  - `pi`: ndarray of shape `(N_s,)`.
  - `log_A`, `log_B`, `log_pi`: ndarrays storing log-space probability matrices.
- Methods:
  - `decode(self, observations: list[int] | np.ndarray) -> tuple[list[int], float]`:
    Executes log-space Viterbi algorithm for observation sequence $\mathbf{O}$.
    Returns tuple `(best_path, log_probability)`:
    - `best_path`: list of $T$ integers representing optimal state indices.
    - `log_probability`: float representing $\ln P(Q^*, \mathbf{O})$.
    Raises `ValueError` if observations are empty or contain invalid indices.

F. **Guided Implementation Stages**
**Step 5: Algorithm Design & Pseudocode**
```text
decode(observations):
  T = len(observations)
  V = zeros(T, N_s)
  BP = zeros(T, N_s, dtype=int)
  
  # Initialization:
  V[0] = log_pi + log_B[:, observations[0]]
  
  # Recursion:
  For t from 1 to T-1:
    o_t = observations[t]
    For j in range(N_s):
      candidates = V[t-1] + log_A[:, j]
      best_prev = argmax(candidates)
      BP[t, j] = best_prev
      V[t, j] = candidates[best_prev] + log_B[j, o_t]
      
  # Termination:
  best_last_state = argmax(V[T-1])
  best_log_prob = V[T-1, best_last_state]
  
  # Backtracking:
  path = [best_last_state]
  For t from T-1 down to 1:
    prev_state = BP[t, path[-1]]
    path.append(prev_state)
  path.reverse()
  return path, best_log_prob
```

**Checkpoint 1: Safe Log-Transformation and Initialization**
- **What to learn**: Safe log conversions handling zero probability transitions without Python warnings.
- **What to do**: In `__init__`, use `np.where(p > 0, np.log(p), -np.inf)` to construct `log_A`, `log_B`, `log_pi`. Implement $t=0$ trellis initialization.
- **How to check yourself**: Initialize with Dataset H1. Verify $V_0 = [\ln(0.36), \ln(0.04)] \approx [-1.02165, -3.21887]$.
- **When to proceed**: Trellis initialization matches hand calculations.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Direct `np.log(0)` issues runtime warnings; suppress or use conditional masking.
  - *Hint 2 (Operation)*: `with np.errstate(divide='ignore'): log_p = np.log(p)`.
  - *Hint 3 (Debugging)*: Check `log_pi.shape == (N_s,)`.

**Checkpoint 2: Dynamic Programming Trellis and Backpointers**
- **What to learn**: Building the forward dynamic programming trellis and recording argmax history.
- **What to do**: Iterate $t \in [1, T-1]$, computing candidates across all states $i \to j$. Store best candidate in `V[t, j]` and predecessor in `BP[t, j]`.
- **How to check yourself**: Compute $t=1$ and $t=2$ on Dataset H1. Verify backpointer array matches hand derivation: $BP[1] = [0, 0]$ and $BP[2] = [0, 1]$.
- **When to proceed**: Entire trellis values match analytical step-by-step numbers.
- **Recovery hints**:
  - *Hint 1 (Concept)*: `candidates = V[t-1, :] + self.log_A[:, j]`.
  - *Hint 2 (Operation)*: Vectorize over states: `candidates = V[t-1, :, None] + self.log_A` has shape `(N_s, N_s)`.
  - *Hint 3 (Debugging)*: Make sure `BP` has `dtype=int`!

**Checkpoint 3: Backtracking and Trajectory Reconstruction**
- **What to learn**: Tracing back through backpointers from $T$ down to $1$.
- **What to do**: Identify best final state at $t=T-1$, then trace backward using `BP[t, curr_state]`.
- **How to check yourself**: On Dataset H1 with $\mathbf{O} = [0, 1, 2]$, `decode` must return `([0, 1, 1], -4.34590)`.
- **When to proceed**: Path exactly recovers the global optimal sequence.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Remember that list append and reverse reconstructs chronologically from $t=1$ to $t=T$.
  - *Hint 2 (Operation)*: `path.reverse()` or slice `path[::-1]`.
  - *Hint 3 (Debugging)*: Check index alignment: $BP[t]$ points from state at time $t$ back to time $t-1$.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_dataset_h1_exact`: Verify that `decode([0, 1, 2])` yields `[0, 1, 1]` and log-probability matching $\ln(0.01296) \approx -4.345905$.
2. `test_long_sequence_underflow`: Test on sequence length $T=1000$. Standard probabilities would underflow to $0.0$; verify log-space Viterbi outputs a valid path and finite negative log-probability.
3. `test_impossible_transition`: Set $A_{0, 1} = 0.0$. Verify that transitions from state $0$ to $1$ are strictly forbidden ($-\infty$).
4. `test_empty_sequence_error`: Confirm passing an empty sequence raises `ValueError`.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Initialization: $O(N_s)$.
  - Trellis recursion: $O(T \cdot N_s^2)$ operations (evaluating transitions between all pairs of states at each step).
  - Backtracking: $O(T)$ operations.
  - Total time: $O(T \cdot N_s^2)$.
- **Space Complexity**:
  - Trellis array `V`: $O(T \cdot N_s)$.
  - Backpointer array `BP`: $O(T \cdot N_s)$.
  - Total auxiliary space: $O(T \cdot N_s)$.
- **Trade-offs**: Finds the exact globally optimal sequence in linear time with respect to sequence length $T$, but requires memory proportional to $T \cdot N_s$.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement Viterbi forward trellis filling for a 2-state HMM with manual printouts.
- **Level 2 (Standard)**: Implement full `HMMViterbi` class with vectorized log-space dynamic programming, backpointer tracking, and backward path reconstruction.
- **Level 3 (Challenge)**: Extend to beam search decoding: at each time step, retain only the top-$K$ most likely states to reduce complexity when state space $N_s$ is very large (e.g. speech recognition).
- **Definition of Done**: Solves Dataset H1 matching hand calculations, passes $T=1000$ without underflow, and handles forbidden transitions.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. How does the Viterbi algorithm relate to Dijkstra's shortest path algorithm on an acyclic directed graph?
2. If two paths have identical maximum log-probabilities, how does `np.argmax` resolve the tie, and does it affect the mathematical validity of the optimal decoding?
"""
