"""
A. **Mission**
While Momentum accelerates SGD, it still applies a global learning rate to all parameters. RMSProp (Root Mean Square Propagation) addresses the issue that different parameters may require significantly different learning rates. It maintains a running average of squared gradients and divides the current gradient by the root of this average, effectively normalising the gradient magnitude. This builds on the limitations of vanilla SGD and Momentum.

B. **Prerequisites**
- SGD exercise (Hard).
- Concept of adaptive learning rates (Helpful).

C. **Learning Questions**
1. Why is it beneficial to scale down the learning rate for parameters with large, consistent gradients and scale it up for those with small gradients?
2. What is the purpose of the `epsilon` term in the denominator?
3. How does the exponential moving average of squared gradients prevent the learning rate from decaying to zero too quickly (unlike Adagrad)?

D. **Mathematics to Derive**
1. Write the update rules for the running average of squared gradients and the parameter update.
2. Explain the scaling effect on the gradient step.

E. **Implementation Contract**
- Class: `RMSProp`
- Constructor: `__init__(self, lr=0.01, beta=0.99, epsilon=1e-8)`
  - `lr`: Base learning rate.
  - `beta`: Decay rate for the moving average.
  - `epsilon`: Small constant for numerical stability.
- Methods:
  - `step(self, params, grads)`: Update parameters.

F. **Guided Implementation Stages**
1. **Initialisation**: Store hyperparameters. Prepare to track the `squared_grad_avg` state (initialised to zeros on the first step).
2. **State Update**: For each parameter, update the moving average: `s = beta * s + (1 - beta) * (grad ** 2)`.
3. **Parameter Update**: Compute the step: `w = w - (lr / (sqrt(s) + epsilon)) * grad`.

G. **Edge Cases and Expected Tests**
- Verify numerical stability (no division by zero) when gradients are exactly zero, thanks to `epsilon`.
- Ensure state variables are initialised correctly matching parameter shapes.
- Test the normalisation effect: a huge gradient should result in a bounded step size.

H. **Complexity Analysis**
- Space Complexity: Compare the memory requirements of RMSProp with Momentum and SGD.

I. **Definition of Done**
- Adaptive learning rate mechanism is correctly implemented.
- The optimiser maintains internal state for squared gradients.

J. **Reflection**
1. RMSProp normalises gradients based on magnitude but ignores direction. Could we combine the directional benefits of momentum with the magnitude normalisation of RMSProp?
"""
