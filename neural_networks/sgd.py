"""
A. **Mission**
Implement Vanilla Stochastic Gradient Descent to optimize the MLP.

B. **Prerequisites**
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization): Gradient descent.

C. **Learning Questions**
1. Why is SGD noisy compared to full batch descent?

D. **Mathematics to Derive**
1. Expected value of stochastic gradient vs full gradient.

E. **Implementation Contract**
- Class: `SGD`
- `__init__(self, lr=0.01)`
- `step(self, params, grads)`

F. **Guided Implementation Stages**
- **Stage 0: Step Function**
  - **What to learn**: Applying gradients.
  - **What to do**: Update params by subtracting lr * grad.
  - **How to check yourself**: Params should change in opposite direction of grad.
  - **When to proceed**: When lists of params update correctly.
  - **Recovery hints 1/2/3**: Zip params and grads; -= operator; check lr scale.

G. **Edge Cases and Expected Tests**
- Fails or is very slow on Dataset 6C (Ill-conditioned quadratic loss surface).

H. **Complexity Analysis**
- Time: O(P) per step, where P is param count.

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Single param update.
- Level 2 Standard: List of params update.
- Level 3 Challenge: Integrate with MLP.
- DoD: Successfully trains MLP on XOR.

J. **Reflection**
What happens if learning rate is too large?
"""
