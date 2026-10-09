"""
A. **Mission**
Implement SGD with Momentum to accelerate convergence on ill-conditioned surfaces (Dataset 6C).

B. **Prerequisites**
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization): Gradients.

C. **Learning Questions**
1. How does velocity damp oscillations?

D. **Mathematics to Derive**
1. Velocity update rule and effective learning rate in consistent directions.

E. **Implementation Contract**
- Class: `Momentum`
- `__init__(self, lr=0.01, beta=0.9)`
- `step(self, params, grads)`

F. **Guided Implementation Stages**
- **Stage 0: Velocity Initialization**
  - **What to learn**: Stateful optimizers.
  - **What to do**: Initialize velocities to zero matching param shapes.
  - **How to check yourself**: len(velocities) == len(params).
  - **When to proceed**: When shapes match.
  - **Recovery hints 1/2/3**: Use np.zeros_like; check on first step call.
- **Stage 1: Velocity Update**
  - **What to learn**: Exponential moving average.
  - **What to do**: v = beta*v - lr*grad.
  - **How to check yourself**: Velocity should grow if grad is constant.
  - **When to proceed**: When step works on toy arrays.
  - **Recovery hints 1/2/3**: Check beta term; update param with v.

G. **Edge Cases and Expected Tests**
- Outperforms SGD on Dataset 6C.

H. **Complexity Analysis**
- Space: O(P) for velocities.

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Single param momentum.
- Level 2 Standard: Full momentum step.
- Level 3 Challenge: Nesterov momentum (optional).
- DoD: Solves Dataset 6C faster than SGD.

J. **Reflection**
Why not set beta to 0.999?
"""
