"""
A. **Mission**
Vanilla SGD can be slow to navigate ravines (valleys in the loss landscape) and tends to oscillate. SGD with Momentum addresses this by accumulating a velocity vector, acting like a heavy ball rolling downhill. It builds upon the SGD exercise by adding a memory of past gradients, which dampens oscillations and accelerates convergence in consistent directions.

B. **Prerequisites**
- SGD exercise (Hard).
- Understanding of exponential moving averages (Helpful).

C. **Learning Questions**
1. How does the momentum term mathematically dampen oscillations in directions of high curvature?
2. Why is the momentum hyperparameter (beta) typically chosen to be close to 1 (e.g., 0.9)?
3. What is the physical analogy of momentum in this context?

D. **Mathematics to Derive**
1. Write down the update equations for velocity and the parameters in SGD with Momentum.
2. Analyze the effective learning rate in a direction where the gradient is constant over many steps.

E. **Implementation Contract**
- Class: `Momentum`
- Constructor: `__init__(self, lr=0.01, beta=0.9)`
  - `lr`: Learning rate (float).
  - `beta`: Momentum coefficient (float).
- Methods:
  - `step(self, params, grads)`: Update parameters using momentum.
    - `params`: List of NumPy arrays.
    - `grads`: List of NumPy arrays.

F. **Guided Implementation Stages**
1. **Initialisation**: Store `lr` and `beta`. Initialise a `velocities` list to store the moving average of gradients. The velocities must match the shapes of `params`.
2. **First Step**: On the first call to `step`, you may need to initialise the velocity arrays to zeros of the correct shapes.
3. **Velocity Update**: For each parameter, update its velocity: `v = beta * v - lr * grad`. 
4. **Parameter Update**: Update the parameter: `w = w + v`.

G. **Edge Cases and Expected Tests**
- Test that velocity accumulation correctly accelerates movement when gradients are constant.
- Ensure velocity arrays are correctly initialised on the first step for arbitrary parameter shapes.
- Test with beta=0 to ensure it behaves exactly like vanilla SGD.

H. **Complexity Analysis**
- Space Complexity: How much extra memory is required compared to vanilla SGD?

I. **Definition of Done**
- Velocity is correctly tracked and applied.
- The momentum update rule matches the physical analogy of accumulating speed.

J. **Reflection**
1. While momentum helps, what happens if the learning rate is uniform across all parameters but some parameters need much larger updates than others?
"""
