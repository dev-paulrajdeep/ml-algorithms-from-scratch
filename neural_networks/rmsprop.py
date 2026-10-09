"""
A. **Mission**
Implement RMSProp to adapt learning rates per parameter using gradient magnitudes.

B. **Prerequisites**
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization): Gradients.

C. **Learning Questions**
1. Why divide by root of squared gradients?

D. **Mathematics to Derive**
1. Exponential moving average of squared gradients and step size.

E. **Implementation Contract**
- Class: `RMSProp`
- `__init__(self, lr=0.01, beta=0.99, epsilon=1e-8)`
- `step(self, params, grads)`

F. **Guided Implementation Stages**
- **Stage 0: State Initialization**
  - **What to learn**: Tracking second moments.
  - **What to do**: Init squared_avg state.
  - **How to check yourself**: np.zeros_like for all params.
  - **When to proceed**: On successful init.
  - **Recovery hints 1/2/3**: Check shapes; do on first step.
- **Stage 1: Step Update**
  - **What to learn**: Gradient normalization.
  - **What to do**: Update avg and params.
  - **How to check yourself**: Avoid div by zero.
  - **When to proceed**: When works on Dataset 6C.
  - **Recovery hints 1/2/3**: Use epsilon; grad**2; np.sqrt.

G. **Edge Cases and Expected Tests**
- Stability with zero gradients (epsilon protects).

H. **Complexity Analysis**
- Space: O(P) for second moments.

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Basic update.
- Level 2 Standard: Epsilon protection.
- Level 3 Challenge: Compare to Adagrad.
- DoD: Succeeds on Dataset 6C.

J. **Reflection**
How does RMSProp differ from Momentum conceptually?
"""
