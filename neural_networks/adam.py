"""
A. **Mission**
Implement Adam, combining Momentum and RMSProp with bias correction.

B. **Prerequisites**
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization): Gradients.

C. **Learning Questions**
1. Why is bias correction needed for early iterations?

D. **Mathematics to Derive**
1. Adam first and second moment bias corrections.

E. **Implementation Contract**
- Class: `Adam`
- `__init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)`
- `step(self, params, grads)`

F. **Guided Implementation Stages**
- **Stage 0: Dual State Initialization**
  - **What to learn**: Managing multiple states.
  - **What to do**: Init m and v arrays.
  - **How to check yourself**: Both are zeros_like params.
  - **When to proceed**: Shapes correct.
  - **Recovery hints 1/2/3**: Two lists; step counter t=0.
- **Stage 1: Bias Corrected Updates**
  - **What to learn**: Fixing zero initialization bias.
  - **What to do**: Update m, v, then m_hat, v_hat.
  - **How to check yourself**: m_hat should be larger than m at t=1.
  - **When to proceed**: Step applies correctly.
  - **Recovery hints 1/2/3**: Increment t; divide by 1-beta**t; update param.

G. **Edge Cases and Expected Tests**
- Correct bias correction at t=1 vs t=100.
- Dominates Dataset 6C.

H. **Complexity Analysis**
- Space: O(2P) for states.

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Updates without bias correction.
- Level 2 Standard: Full Adam with bias correction.
- Level 3 Challenge: AdamW (weight decay).
- DoD: Successfully scales learning dynamically.

J. **Reflection**
Why is Adam the default for many modern deep learning tasks?
"""
