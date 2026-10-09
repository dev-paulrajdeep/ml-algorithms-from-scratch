"""
A. **Mission**
Adam (Adaptive Moment Estimation) is one of the most popular optimisers in deep learning. It combines the heuristics of both Momentum (first moment, tracking direction) and RMSProp (second moment, tracking magnitude). Additionally, Adam includes bias correction mechanisms to counteract the fact that these moving averages are initialised to zero, which heavily biases early iterations. This is the culmination of the optimiser sequence.

B. **Prerequisites**
- Momentum exercise (Hard).
- RMSProp exercise (Hard).

C. **Learning Questions**
1. How does Adam conceptually combine the strengths of Momentum and RMSProp?
2. Why is bias correction crucial, especially in the first few iterations of training?
3. In what scenarios might Adam fail or perform worse than well-tuned SGD with Momentum?

D. **Mathematics to Derive**
1. Write the full set of equations for Adam: first moment update, second moment update, bias corrections, and final parameter update.
2. Prove why the bias correction term `(1 - beta^t)` is necessary given zero initialisation of the moments.

E. **Implementation Contract**
- Class: `Adam`
- Constructor: `__init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)`
- Methods:
  - `step(self, params, grads)`: Update parameters.

F. **Guided Implementation Stages**
1. **Initialisation**: Store hyperparameters and set up `m` (first moment) and `v` (second moment) lists, along with a step counter `t` starting at 0.
2. **Step Increment**: Increment `t` by 1 on each call to `step()`.
3. **Moment Updates**:
   - `m = beta1 * m + (1 - beta1) * grad`
   - `v = beta2 * v + (1 - beta2) * (grad ** 2)`
4. **Bias Correction**:
   - `m_hat = m / (1 - beta1 ** t)`
   - `v_hat = v / (1 - beta2 ** t)`
5. **Parameter Update**:
   - `w = w - (lr / (sqrt(v_hat) + epsilon)) * m_hat`

G. **Edge Cases and Expected Tests**
- Verify the effect of bias correction by comparing updates at `t=1` with and without it.
- Ensure robust handling of zero gradients.
- Confirm state memory aligns perfectly with varying parameter array shapes.

H. **Complexity Analysis**
- Space Complexity: How many copies of the parameters must Adam store in memory?

I. **Definition of Done**
- First and second moments are correctly tracked and updated.
- Bias correction is applied accurately using the time step `t`.
- The final update perfectly synthesises Momentum and RMSProp concepts.

J. **Reflection**
1. Why has Adam become the default optimiser for many deep learning practitioners, despite the theoretical guarantees of SGD?
"""
