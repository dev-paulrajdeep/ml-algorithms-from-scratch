"""
A. **Mission**
Gradient Boosting is a sequential ensemble that builds trees iteratively, where each new tree fits the negative gradient (residuals) of the loss function, reducing bias.

B. **Prerequisites**
- [Stage 0F: ML Core](../foundations/README.md#module-0f-machine-learning-core-concepts) (bias-variance tradeoff)
- Decision Tree (Regression mode)

C. **Learning Questions**
1. How does Boosting differ from Bagging?
2. What is the role of learning rate (shrinkage)?
3. Why do we always use Regression Trees as base learners?

D. **Mathematics to Derive**
1. Compute negative gradient of MSE loss.
2. Boosting gradient update derivation.

E. **Implementation Contract**
- Implement `class GradientBoosting`.
- Methods:
  - `__init__(self, task='regression', n_estimators=100, learning_rate=0.1, max_depth=3)`
  - `fit(self, X, y)`: Fit sequence of trees.
  - `predict(self, X)`: Return aggregated predictions.

F. **Guided Implementation Stages**
- **Step 0**: Base Initialization.
  - *What to learn*: Starting point for boosting.
  - *What to do*: Initialize with mean of `y`.
  - *How to check yourself*: Prediction without trees is just the mean.
  - *When to proceed*: Base predictor works.
  - *Recovery hints 1*: Store it as $F_0$.
- **Step 1**: Compute Gradients.
  - *What to learn*: Calculating pseudo-residuals.
  - *What to do*: Implement residual calculation for MSE.
  - *How to check yourself*: Residuals sum to 0 after first step.
  - *When to proceed*: Correct residuals.
  - *Recovery hints 1*: $r = y - F$.
- **Step 2**: Fit Weak Learner.
  - *What to learn*: Fitting trees to residuals.
  - *What to do*: Fit a DecisionTree (regression) to $r$.
  - *How to check yourself*: Tree output resembles residual shape.
  - *When to proceed*: Tree fits.
  - *Recovery hints 1*: Ensure task is always regression.
- **Step 3**: Update Model.
  - *What to learn*: Shrinkage.
  - *What to do*: Update predictions with tree scaled by `learning_rate`.
  - *How to check yourself*: Error decreases iteratively.
  - *When to proceed*: Full loop complete.
  - *Recovery hints 1*: $F_m = F_{m-1} + \nu \cdot h_m$.
- **Step 4**: Prediction.
  - *What to learn*: Sequential inference.
  - *What to do*: Implement `predict` by summing base and all trees.
  - *How to check yourself*: Test on Dataset 3C.
  - *When to proceed*: Test passes.
  - *Recovery hints 1*: Remember to multiply by learning rate.

G. **Edge Cases and Expected Tests**
- `learning_rate=0` predicts initial mean.
- Extreme overfitting with `max_depth=None` and `learning_rate=1`.

H. **Complexity Analysis**
1. Training time complexity? Can it be parallelized?

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Implement regression with MSE.
- Level 2 Standard: Ensure learning rate is correctly applied.
- Level 3 Challenge: Implement for binary classification (Log Loss).
- Done: Outperforms single tree on Dataset 3C.

J. **Reflection**
- Why weak learners (shallow trees) for boosting?
"""
