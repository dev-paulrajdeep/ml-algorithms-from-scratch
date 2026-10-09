"""
A. **Mission**
The Forward-Backward algorithm solves the "evaluation" problem for Hidden Markov Models (computing the probability of an observation sequence) and, crucially, computes the posterior probability of being in a specific state at a specific time given the entire observation sequence. This algorithm is the E-step of the Baum-Welch algorithm, which learns HMM parameters via Expectation-Maximization.

B. **Prerequisites**
- Probability theory (Bayes' theorem, conditional probability, joint distributions, marginalization).
- Maximum Likelihood Estimation (MLE) and iterative optimization.
- Expectation-Maximization (EM) concepts.
- Understanding of latent-variable models.

C. **Learning Questions**
1. What is the difference between the forward probability $\\alpha_t(i)$ and the backward probability $\\beta_t(i)$ in terms of the observations they condition on?
2. How do we combine the forward and backward probabilities to find the posterior probability of being in state $i$ at time $t$?
3. Why is scaling or log-space computation even more critical in the Forward-Backward algorithm compared to Viterbi?
4. How does Baum-Welch utilize the state posteriors and transition posteriors to update the HMM parameters?

D. **Mathematics to Derive**
1. Derive the forward recurrence: $\\alpha_{t}(j) = P(o_1, \\dots, o_t, q_t=j | \\lambda)$ in terms of $\\alpha_{t-1}(i)$.
2. Derive the backward recurrence: $\\beta_{t}(i) = P(o_{t+1}, \\dots, o_T | q_t=i, \\lambda)$ in terms of $\\beta_{t+1}(j)$.
3. Show how to calculate the marginal probability $P(O|\\lambda)$ using just the forward variables, and then verify it using just the backward variables.
4. Derive the Baum-Welch update formulas for the transition probabilities $A_{ij}$ and emission probabilities $B_{jk}$ based on expected counts.

E. **Implementation Contract**
- Class name: `HMMForwardBackward`
- Initialization: `__init__(n_states, n_observations)`
- Attributes: `transition_` (shape `(n_states, n_states)`), `emission_` (shape `(n_states, n_observations)`), `initial_` (shape `(n_states,)`).
- `forward(observations)`: Computes and returns the forward probabilities (or scaled versions) and the log marginal likelihood of the sequence.
- `backward(observations)`: Computes and returns the backward probabilities (or scaled versions).
- `fit(observations_list, max_iters=100, tol=1e-4)`: Implements the Baum-Welch (EM) algorithm to learn the parameters from a list of observation sequences.
- Note: Use log-space or dynamic scaling factors (e.g., normalizing $\\alpha$ at each time step) to prevent underflow.

F. **Guided Implementation Stages**
1. **Initialization**: Initialize `transition_`, `emission_`, and `initial_` probabilities randomly, ensuring each row sums to 1.
2. **Forward Pass**: Implement `forward(observations)`. At each step $t$, compute $\\alpha_t$ and introduce a scale factor $c_t = 1 / \\sum_i \\alpha_t(i)$ to normalize $\\alpha_t$. Store $c_t$ as it's needed for the backward pass and log-likelihood computation.
3. **Backward Pass**: Implement `backward(observations)`. Use the same scale factors $c_t$ from the forward pass to scale $\\beta_t$.
4. **Posteriors (E-step)**: Using scaled $\\alpha$ and $\\beta$, compute the state responsibilities $\\gamma_t(i)$ and transition responsibilities $\\xi_t(i, j)$ for all sequences.
5. **Updates (M-step)**: Update the `initial_`, `transition_`, and `emission_` matrices using the aggregated expected counts from the E-step.
6. **Convergence Loop**: Repeat the E and M steps in `fit()` until the change in the overall log-likelihood across sequences is less than `tol`.

G. **Edge Cases and Expected Tests**
1. Pass a set of completely random sequences and ensure the log-likelihood monotonically increases during `fit()`.
2. Construct a deterministic transition and emission scenario and test whether `forward` correctly computes the expected sequence probability.
3. Train on multiple sequences of varying lengths. Ensure your algorithm gracefully handles length variations.
4. Test with scaling enabled to ensure that sequence lengths of $T > 1000$ do not cause underflow to zero.

H. **Complexity Analysis**
1. What is the time complexity of running Baum-Welch for $M$ sequences of average length $T$, with $N$ states, for $K$ iterations?
2. What are the memory requirements for storing $\\alpha, \\beta, \\gamma, \\xi$ during a single sequence's E-step?

I. **Definition of Done**
- `HMMForwardBackward` successfully fits synthetic discrete-observation sequences and learns meaningful transition and emission distributions.
- The `forward` and `backward` methods properly integrate scaling to prevent underflow.
- `fit()` strictly increases or plateaus the log-likelihood of the training sequences over iterations.

J. **Reflection**
1. Why is Baum-Welch an instance of the Expectation-Maximization algorithm, and what exactly are the latent variables it infers?
2. In practice, how would you handle continuous observations (e.g., speech signals) instead of discrete ones in this framework?
"""
