"""
A. **Mission**
Gradient Boosting is a powerful sequential ensemble technique. Instead of building independent trees like Random Forest, it builds trees one at a time, where each new tree tries to correct the errors made by the combination of all previous trees. It does this by fitting the new tree to the negative gradient (often the residuals) of the loss function. It is a highly effective method that often wins machine learning competitions on tabular data.

B. **Prerequisites**
- Solid understanding of Decision Trees, specifically for regression.
- Calculus: partial derivatives and gradients.
- Understanding of loss functions (e.g., Mean Squared Error, Log Loss).
- Concept of Gradient Descent optimization.

C. **Learning Questions**
1. How does Gradient Boosting differ fundamentally from Bagging (Random Forest)?
2. In Gradient Boosting for regression with MSE loss, what exactly is the target variable for the $m$-th tree?
3. What is the role of the learning rate (shrinkage) in Gradient Boosting, and why is it important for generalization?
4. How is Gradient Boosting a form of gradient descent in "function space"?

D. **Mathematics to Derive**
1. Define the Mean Squared Error loss function $L(y, F(x)) = \frac{1}{2}(y - F(x))^2$. Compute the negative partial derivative of $L$ with respect to the model prediction $F(x)$.
2. Show that fitting a regression tree to the negative gradient of the MSE loss is equivalent to fitting the tree to the residuals $y - F(x)$.
3. For binary classification, let the model output $F(x)$ be the log-odds, and the prediction be $p = \frac{1}{1 + e^{-F(x)}}$. Using the log loss (cross-entropy), compute the negative gradient with respect to $F(x)$. What does this negative gradient look like?

E. **Implementation Contract**
- Implement `class GradientBoosting`.
- Methods:
  - `__init__(self, task='regression', n_estimators=100, learning_rate=0.1, max_depth=3)`
    - `task`: 'regression' or 'classification'. (Focus on regression with MSE loss first, then optionally classification).
    - `n_estimators`: Number of boosting stages to perform.
    - `learning_rate`: Shrinkage parameter controlling the contribution of each tree.
    - `max_depth`: Maximum depth of the individual regression trees.
  - `fit(self, X, y)`: Fit the sequence of trees.
  - `predict(self, X)`: Predict targets for X.
- You MUST use your `DecisionTree` implementation. Crucially, the base learners in Gradient Boosting are *always* Regression Trees, even for classification tasks (because they predict continuous gradients/updates).

F. **Guided Implementation Stages**
1. **Base Initialization**: Initialize the model prediction $F_0(x)$. For regression with MSE, a common initialization is simply the mean of the training targets $y$.
2. **Boosting Loop**: Create a loop for `n_estimators` iterations.
3. **Compute Residuals (Gradients)**: In each iteration $m$, compute the negative gradients (pseudo-residuals) $r_{im} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$. For MSE, this is simply $y_i - F_{m-1}(x_i)$.
4. **Fit Weak Learner**: Fit a `DecisionTree` (task='regression', max_depth=`max_depth`) to the features `X` and targets `r_m`.
5. **Update Model**: Update the current predictions: $F_m(X) = F_{m-1}(X) + \nu \cdot \text{tree}_m(X)$, where $\nu$ is the `learning_rate`. Store the trained tree.
6. **Prediction**: To predict on new data, start with the initial value $F_0$ and add the predictions of all stored trees multiplied by the learning rate.
7. **Classification Extension (Optional/Advanced)**: Implement the logistic loss gradient and update rule for binary classification, ensuring predictions are converted back to probabilities via the sigmoid function.

G. **Edge Cases and Expected Tests**
- *Zero Learning Rate*: A learning rate of 0.0 should result in predictions exactly equal to the initial guess (e.g., mean of `y`), regardless of `n_estimators`.
- *Overfitting*: With `learning_rate=1.0` and high `n_estimators`, the model should perfectly fit the training data (and likely overfit).
- *Base Learner Usage*: Ensure that `DecisionTree` is called with `task='regression'` even if `GradientBoosting` is set to `task='classification'`.

H. **Complexity Analysis**
1. What is the training time complexity compared to Random Forest? Can the boosting loop be parallelized?
2. If we use very deep trees (e.g., `max_depth=None`) in Gradient Boosting, what happens to the training error after just a few iterations?

I. **Definition of Done**
- GradientBoosting class correctly implements sequential fitting.
- Model successfully minimizes MSE on a regression dataset.
- `predict` correctly aggregates the base prediction and the scaled tree predictions.
- Shrinkage (`learning_rate`) is applied correctly.

J. **Reflection**
- Why are shallow trees (stumps or depth 2-3) typically preferred as base learners in Gradient Boosting, whereas deep trees are preferred in Random Forests?
- If your Gradient Boosting model is overfitting, what two hyperparameters are the most direct ways to regularize it?
"""
