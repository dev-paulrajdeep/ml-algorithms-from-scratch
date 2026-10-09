"""
A. **Mission**
The Linear Support Vector Machine (SVM) aims to find the optimal hyperplane that separates classes with the maximum margin using hinge loss and subgradient descent.

B. **Prerequisites**
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization)

C. **Learning Questions**
1. What is a margin?
2. What role do support vectors play?

D. **Mathematics to Derive**
1. Derive the subgradient of the hinge loss objective.

E. **Implementation Contract**
- Class: `LinearSVM`
- `__init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000)`
- `fit(self, X, y)`: Train the SVM. `X` is (n_samples, n_features), `y` is (n_samples,) with -1/+1 labels.
- `predict(self, X)`: Predict class labels (-1 or 1).
- Allow NumPy.

F. **Guided Implementation Stages**
Step 1: Label Verification
- **What to learn**: SVM conventions.
- **What to do**: Ensure `y` contains -1 and 1.
- **How to check yourself**: `np.unique(y)`
- **When to proceed**: Labels are correct.
- **Recovery hints 1/2/3**: Convert 0 to -1 if necessary.

Step 2: Initialization
- **What to learn**: Setup for optimization.
- **What to do**: Initialize weights and bias.
- **How to check yourself**: Shapes match features.
- **When to proceed**: Initialized.
- **Recovery hints 1/2/3**: Zeros are fine.

Step 3: Hinge Loss Subgradient
- **What to learn**: Piecewise derivatives.
- **What to do**: In a loop, compute `condition = y_i * (w^T x_i - b) >= 1`. Update weights based on this condition (L2 penalty only vs penalty + hinge gradient).
- **How to check yourself**: Ensure correct signs in updates.
- **When to proceed**: Can compute updates for single samples or vectorized.
- **Recovery hints 1/2/3**: For condition False, `dw += -y_i * x_i`.

Step 4: Gradient Descent Loop
- **What to learn**: Full optimization.
- **What to do**: Integrate the step update into the `num_iterations` loop.
- **How to check yourself**: Loss decreases.
- **When to proceed**: Converges on linearly separable data.
- **Recovery hints 1/2/3**: Tune learning rate if diverging.

G. **Edge Cases and Expected Tests**
1. `test_margin_violation`: Points inside the margin contribute to the gradient.
2. `test_linear_separability`: Finds separating hyperplane on simple 2D data.

H. **Complexity Analysis**
1. What is the time complexity of training?

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Loop over samples for gradients.
- **Level 2 (Standard)**: Vectorize the condition check and gradient computation.
- **Level 3 (Challenge)**: Add early stopping based on validation loss.
- **Definition of Done**: Tests pass, minimizes hinge loss.

J. **Reflection**
1. Why prefer SVM over Logistic Regression for certain datasets?
"""
