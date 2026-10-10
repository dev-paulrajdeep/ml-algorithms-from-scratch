r"""
A. **Mission** (Step 0: Understand the Problem)
Gradient Boosting is a sequential ensemble learning technique that builds models additively: each new base learner (typically a shallow regression tree) is trained to predict the **pseudo-residuals** (negative gradients of the loss function) of the current ensemble.
Real-world problem: State-of-the-art predictive accuracy on structured tabular datasets (the underlying philosophy behind XGBoost, LightGBM, and CatBoost).
Unlike Random Forest (which averages independent deep trees in parallel to reduce variance), Gradient Boosting iteratively trains shallow "weak" learners to correct remaining mistakes, performing **gradient descent directly in function space** to reduce bias.

B. **Prerequisites**
- [Decision Tree](./decision_tree.py) — Must understand regression trees (variance reduction) as the base learner.
- [Chapter 1: Regression](../regression/README.md) — Residuals ($y - \hat{y}$) and gradient descent parameter updates.
- [Stage 0D: Calculus & Optimization](../foundations/README.md#module-0d-calculus-and-optimization) — Gradients of loss functions.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Additive models, learning rate / shrinkage.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why does the negative gradient of the Mean Squared Error loss function $-\frac{\partial L}{\partial \hat{y}}$ equal the familiar residual $y - \hat{y}$?
2. Why does Gradient Boosting perform best when base learners are shallow trees (e.g., `max_depth = 3`), whereas Random Forest relies on deep unpruned trees?
3. What is the role of the learning rate $\nu$ (shrinkage)? Why does scaling each tree's contribution by a small number (e.g., $0.1$) drastically improve generalization?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider 3 points: $x = [1, 2, 3]$, true continuous targets $y = [10.0, 20.0, 30.0]$.
   - Step 0: Baseline constant prediction $F_0(x) = \bar{y} = \frac{10 + 20 + 30}{3} = 20.0$.
   - Initial residuals: $r^{(0)} = y - F_0 = [10 - 20, 20 - 20, 30 - 20] = [-10.0, 0.0, +10.0]$.
   - Fit weak learner $h_1(x)$ to predict residuals. Suppose tree predicts $[-8.0, 0.0, +8.0]$.
   - Apply shrinkage $\nu = 0.5$:
     $F_1(x) = F_0(x) + \nu \cdot h_1(x) = 20.0 + 0.5 \cdot [-8, 0, 8] = [16.0, 20.0, 24.0]$.
   - New residuals: $r^{(1)} = y - F_1 = [10 - 16, 20 - 20, 30 - 24] = [-6.0, 0.0, +6.0]$.
   - Notice: Error magnitude dropped from $10$ to $6$!
2. **Step 3: Mathematical Notation**:
   - $M$: number of boosting stages (`n_estimators`).
   - $\nu \in (0, 1]$: learning rate / shrinkage parameter.
   - $F_m(\mathbf{x})$: ensemble prediction at stage $m$.
   - $h_m(\mathbf{x})$: weak learner (regression tree) trained at stage $m$.
   - $L(y, F(\mathbf{x}))$: loss function (e.g., $\frac{1}{2}(y - F(\mathbf{x}))^2$).
3. **Step 4: Derive the Functional Gradient Descent Update**:
   - In standard gradient descent, we update parameters: $\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla_{\mathbf{w}} L$.
   - In functional gradient descent, we update the entire function $F(\mathbf{x})$:
     $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) - \nu \cdot \left[ \frac{\partial L(y, F(\mathbf{x}))}{\partial F(\mathbf{x})} \right]_{F = F_{m-1}}$$
   - For Mean Squared Error loss $L(y, F) = \frac{1}{2} (y - F)^2$:
     $$-\frac{\partial L}{\partial F} = -(F - y) = y - F$$
   - Therefore, the negative gradient is simply the residual!
   - At each stage $m$, we fit $h_m(\mathbf{x})$ to targets $r_{im} = y_i - F_{m-1}(\mathbf{x}_i)$, and update:
     $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x})$$

E. **Implementation Contract**
- Class: `GradientBoosting`
- Constructor:
  - `__init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, task: str = 'regression')`
- Methods:
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Initializes $F_0 = \bar{y}$. Sequentially fits $M$ regression `DecisionTree` instances to residuals, storing them in `trees_`.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Computes $F_0 + \nu \sum_{m=1}^M h_m(\mathbf{X})$. Returns 1D array of shape `(N,)`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X, y):
  F_0 = mean(y) # initial constant prediction
  F = full_like(y, F_0)
  trees_ = []
  
  For m in range(n_estimators):
    residuals = y - F
    tree = DecisionTree(task='regression', max_depth=max_depth)
    tree.fit(X, residuals)
    trees_.append(tree)
    
    # Update current predictions with shrinkage
    F += learning_rate * tree.predict(X)

predict(X):
  preds = full(X.shape[0], F_0)
  For tree in trees_:
    preds += learning_rate * tree.predict(X)
  return preds
```

**Checkpoint 1: Initial Base Prediction**
- **What to learn**: Optimal starting value for MSE loss.
- **What to do**: Store `self.F_0_ = np.mean(y)`.
- **How to check yourself**: If $M=0$, prediction is the scalar mean of training targets.
- **When to proceed**: Baseline prediction is recorded and matches $\bar{y}$.
- **Recovery hints**:
  - *Hint 1 (Concept)*: The constant that minimizes MSE is mathematically the sample mean.
  - *Hint 2 (Operation)*: `self.F_0_ = float(np.mean(y))`.
  - *Hint 3 (Debugging)*: Check `assert np.isscalar(self.F_0_)`.

**Checkpoint 2: Computing Residuals & Fitting Weak Learners**
- **What to learn**: Training regression trees on error vectors.
- **What to do**: In each stage, compute `residuals = y - F`, then fit a shallow `DecisionTree(task='regression', max_depth=max_depth)`.
- **How to check yourself**: Residuals on the first step sum to approximately $0.0$.
- **When to proceed**: Each tree captures the dominant remaining error pattern.
- **Recovery hints**:
  - *Hint 1 (Concept)*: The target for `tree.fit(X, residuals)` is the residual array, NOT the original $y$.
  - *Hint 2 (Operation)*: Ensure `DecisionTree` is in regression mode (`criterion='variance'`).
  - *Hint 3 (Debugging)*: Keep `max_depth` small (2 or 3) to prevent base trees from overfitting early residuals.

**Checkpoint 3: Shrinkage and Ensemble Accumulation**
- **What to learn**: Combining contributions across the additive sequence.
- **What to do**: Update training predictions `F += learning_rate * tree.predict(X)` and verify that training MSE decreases after each added tree.
- **How to check yourself**: Plot or print MSE per stage: it must decrease monotonically on clean data.
- **When to proceed**: Model successfully fits non-linear curves on Dataset T3.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Shrinkage acts as a step size in gradient descent.
  - *Hint 2 (Operation)*: In `predict`, start with `np.full(X.shape[0], self.F_0_)` and add `self.learning_rate * tree.predict(X)`.
  - *Hint 3 (Debugging)*: If predictions diverge, reduce `learning_rate` to $0.05$.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_zero_estimators`: When `n_estimators = 0`, predictions for all query points equal $\bar{y}$.
2. `test_monotonic_loss_decrease`: Verify that each successive tree strictly reduces or maintains training MSE.
3. `test_learning_rate_tradeoff`: Show that a smaller learning rate ($\nu = 0.01$) requires more estimators to reach the same training loss as $\nu = 0.5$, but achieves lower test error on noisy data.
4. `test_overfitting_with_deep_trees`: Show that setting `max_depth = 10` causes the ensemble to overfit rapidly compared to `max_depth = 2`.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Training: $O(M \cdot N \log N \cdot D \cdot \text{depth})$. Crucially, training is **inherently sequential**: Tree $m$ depends on the residuals of Tree $m-1$ and cannot be parallelized across trees.
  - Inference: $O(M \cdot \text{depth})$ per sample.
- **Space Complexity**: $O(M \cdot \text{nodes per tree})$ auxiliary memory.
- **Trade-offs**: Extremely high predictive accuracy and handles heterogeneous feature types, but sensitive to hyperparameters and cannot be parallelized across stages during training.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Boost 3 depth-1 stumps by hand on 1D continuous data, calculating residuals and updates on paper.
- **Level 2 (Standard)**: Complete `GradientBoosting` regression class supporting configurable `n_estimators`, `learning_rate`, and `max_depth`.
- **Level 3 (Challenge)**: Extend Gradient Boosting to **Binary Classification** using Log-Loss (Negative Binomial Log-Likelihood), converting predictions through the logistic sigmoid and updating probability residuals $r = y - p$.
- **Definition of Done**: Demonstrates sequential reduction of residuals, integrates with `DecisionTree`, and outperforms a single tree on Dataset T3.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does Gradient Boosting build trees sequentially to reduce bias, while Random Forest builds trees in parallel to reduce variance?
2. What happens if you set `learning_rate = 1.0` in Gradient Boosting? Why does high shrinkage lead to better generalization?
"""
