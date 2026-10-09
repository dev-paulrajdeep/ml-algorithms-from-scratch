"""
A. **Mission**
A Hidden Markov Model (HMM) is a sequence model that assumes an unobservable (hidden) Markov process generates a sequence of observations. The Viterbi algorithm solves the "decoding" problem: given a sequence of observations and a known HMM (transition and emission probabilities), it finds the most likely sequence of hidden states that produced those observations. It relies on dynamic programming over a trellis structure.

B. **Prerequisites**
- Probability theory (Bayes' theorem, conditional probability, joint distributions).
- Markov chains and the Markov property.
- Dynamic programming concepts.
- Understanding of log-space probabilities for numerical stability.

C. **Learning Questions**
1. Why can't we simply choose the most likely state independently at each time step to find the most likely sequence?
2. How does the Viterbi algorithm use dynamic programming to reduce the search space of possible state sequences from exponential to linear time?
3. What happens if an observation sequence contains a symbol with a zero emission probability from all states?
4. How do the three central HMM problems (Evaluation, Decoding, Learning) differ from each other?

D. **Mathematics to Derive**
1. Formulate the dynamic programming recurrence relation for the Viterbi algorithm. Let $V_{t, k}$ be the maximum probability of a state sequence ending in state $k$ at time $t$. Express $V_{t, k}$ in terms of $V_{t-1, j}$, transition probabilities $A_{j, k}$, and emission probabilities $B_{k, O_t}$.
2. Write the base case $V_{1, k}$ at time $t=1$.
3. Convert the Viterbi recurrence relation from regular probability space to log-probability space. Show how this turns multiplications into additions.

E. **Implementation Contract**
- Class name: `HMMViterbi`
- Initialization: `__init__(transition_probs, emission_probs, initial_probs)` where inputs are NumPy arrays or lists representing probabilities.
- `decode(observations)`: Takes a list or array of observation indices. Returns a tuple `(best_path, log_probability)`.
- `best_path`: A list of integer state indices representing the most likely sequence of hidden states.
- `log_probability`: A float representing the log-probability of the `best_path` sequence.
- Note: Use log-space computation internally to prevent underflow.

F. **Guided Implementation Stages**
1. **Log Transformation**: In `__init__`, convert all input probability matrices to log-probabilities (handle zeros by converting them to `-inf`).
2. **Initialization Step**: For $t=1$, compute the initial path probabilities for each state using `initial_probs` and the emission probability of the first observation. Keep track of backpointers (though empty at $t=1$).
3. **Recursion Step**: Loop through time $t=2$ to $T$. For each state, compute the maximum log-probability of reaching that state from all possible previous states. Store this value and the state index that achieved the maximum (the backpointer).
4. **Termination Step**: At time $T$, find the state with the highest log-probability. This is the end of the most likely path.
5. **Path Backtracking**: Trace back through the backpointers starting from the best final state to reconstruct the most likely sequence.

G. **Edge Cases and Expected Tests**
1. Write a test where an observation is impossible (zero probability) under a certain state, ensuring the path never visits that state.
2. Test with a sequence of identical observations where transitioning to the same state has a very low probability, forcing the most likely path to alternate states.
3. Test with a long sequence (e.g., 1000 observations) to guarantee that log-space computation successfully prevents numerical underflow (the standard probability would be indistinguishable from 0).

H. **Complexity Analysis**
1. What is the time complexity of the Viterbi algorithm in terms of the number of states $N$ and sequence length $T$?
2. What is the space complexity to store the Viterbi trellis and backpointers?

I. **Definition of Done**
- `HMMViterbi` correctly identifies the optimal state sequence for a given HMM and observation sequence.
- Internally uses log-space probabilities and avoids Python's `math.log(0)` errors.
- The returned `log_probability` accurately matches the joint log-probability of the path and the observations.

J. **Reflection**
1. Could the Viterbi algorithm be used to decode sequences if the Markov property did not hold (e.g., transition depended on the last two states)? How would the state space change?
2. Why is dynamic programming specifically suited for this problem over greedy search or exhaustive search?
"""
