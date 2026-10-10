# Chapter 3: Trees and Ensembles

Welcome to Chapter 3! In this chapter, we transition from linear and parametric models to **non-parametric, recursive partitioning models**. You will learn how decision trees recursively split feature space into rectangular regions, how bagging reduces model variance in Random Forests, and how boosting sequentially minimizes residuals via gradient descent in function space.

---

## Chapter Objectives
- Understand recursive binary partitioning of feature space.
- Differentiate between splitting criteria for classification (Gini impurity, Entropy) and regression (Variance reduction / MSE).
- Implement recursive tree growth, base stopping conditions, and leaf node predictions.
- Understand the bias-variance tradeoff of individual deep decision trees (high variance, low bias).
- Implement **Bootstrap Aggregation (Bagging)** and feature subsampling to build a Random Forest that reduces variance.
- Understand **Gradient Boosting** as sequential residual fitting and functional gradient descent that reduces bias.

---

## Prerequisites
Before tackling this chapter, ensure you have completed:
- **[Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics)** — Sample variance, entropy, and expected value.
- **[Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts)** — Underfitting, overfitting, and the bias-variance tradeoff.
- **[Chapter 1: Regression](../regression/README.md)** — Residuals ($y - \hat{y}$) and gradient descent intuition.
- **[Chapter 2: Classification](../classification/README.md)** — Classification accuracy and majority voting.

---

## 🧭 Pedagogical Progression & Intuition Bridges

To build trees and ensembles systematically, absorb these intuitions:

### 1. Before Decision Trees: The Single Manual Split
- Consider 4 points on a number line: $x = [1, 2, 5, 6]$ with labels $y = [0, 0, 1, 1]$.
- A single threshold at $x = 3.5$ perfectly separates the classes. The left bucket $\{0, 0\}$ is $100\%$ pure; the right bucket $\{1, 1\}$ is $100\%$ pure.
- **Impurity**:
  - Gini Impurity: $G = 1 - \sum p_k^2$. For a pure node (all class 0), $G = 1 - (1)^2 = 0$. For an evenly split binary node, $G = 1 - (0.5^2 + 0.5^2) = 0.5$.
  - Entropy: $H = -\sum p_k \log_2(p_k)$. For a pure node, $H = 0$. For an evenly split binary node, $H = 1.0$.
  - Variance Reduction (Regression): Reduction in target variance after splitting continuous $y$.
- **Recursive Splitting**: If a child node is not pure, repeat the splitting procedure on that subset until stopping rules (e.g., `max_depth`, `min_samples_split`) are met.

### 2. Before Random Forest: Bagging Reduces Variance
- A single unpruned decision tree fits training data with $100\%$ accuracy, but it has **high variance**—change one training point and the tree structure completely changes.
- **Ensemble Averaging Principle**: If we average $M$ independent models with variance $\sigma^2$, the variance of the average is $\frac{\sigma^2}{M}$.
- To create diverse, decorrelated trees:
  1. **Bootstrap Aggregating (Bagging)**: Train each tree on a random sample of size $N$ drawn *with replacement* from the training set. (~63.2% unique samples per tree).
  2. **Feature Subsampling**: At each split, evaluate only a random subset of $\sqrt{D}$ features.

### 3. Before Gradient Boosting: Sequential Residual Correction
- Unlike Random Forest (which builds trees independently in parallel to reduce variance), Gradient Boosting builds trees **sequentially to reduce bias**.
- Start with a baseline constant prediction $F_0(x) = \bar{y}$.
- Compute residuals $r_i = y_i - F_0(x_i)$.
- Train a shallow tree $h_1(x)$ to predict the *residuals*.
- Update the ensemble: $F_1(x) = F_0(x) + \nu \cdot h_1(x)$, where $\nu$ is the shrinkage learning rate.
- Repeat! Notice that for MSE loss $\frac{1}{2}(y - \hat{y})^2$, the negative gradient is exactly the residual: $-\frac{\partial L}{\partial \hat{y}} = y - \hat{y}$. Boosting is literally **gradient descent in function space**!

---

## Recommended Study Sequence

1. **[`decision_tree.py`](./decision_tree.py)** — Start here. Build the recursive CART tree supporting Gini/entropy classification and variance-reduction regression.
2. **[`random_forest.py`](./random_forest.py)** — Combine multiple decision trees via bootstrap aggregation and random feature subsampling.
3. **[`gradient_boosting.py`](./gradient_boosting.py)** — Build a sequential residual-fitting ensemble using shallow regression trees.

---

## Chapter 3 Toy Datasets

- **Dataset T1 (1D Single Obvious Split)**:
  - $\mathbf{X} = [[1.0], [2.0], [5.0], [6.0]]$, $\mathbf{y} = [0, 0, 1, 1]$
  - Hand check: Split at threshold $t = 3.5$ on feature 0 yields pure children with Information Gain $= 0.5$.

- **Dataset T2 (2D Axis-Aligned Step Function)**:
  - $\mathbf{X} = [[1, 1], [1, 5], [5, 1], [5, 5]]$, $\mathbf{y} = [0, 0, 1, 1]$
  - Hand check: Feature 0 at $t = 3.0$ cleanly separates class 0 from class 1.

- **Dataset T3 (Noisy Sine Curve for Ensembles)**:
  - $x \in [0, 2\pi]$ with Gaussian noise added to $y = \sin(x)$.
  - Demonstrates how a single deep tree overfits the noise, while a Random Forest averages out the variance to recover the smooth sine wave.

---

## Completion Checklist
- [ ] Calculate Gini impurity and Entropy by hand for $[0, 0, 0, 1]$.
- [ ] Implement `decision_tree.py` (Classification & Regression).
- [ ] Implement `random_forest.py` (Bagging + Feature Subsampling).
- [ ] Implement `gradient_boosting.py` (Residual fitting + Shrinkage).
- [ ] Compare Random Forest vs single Decision Tree variance on Dataset T3.

---

## Cross-Chapter Conceptual Questions
1. **Axis-Aligned Limitations**: Why do decision trees struggle with diagonal decision boundaries (e.g., $x_1 + x_2 > 2$), producing a "staircase" approximation?
2. **Weak vs Strong Learners**: Why does Gradient Boosting perform best with shallow "weak" trees (e.g., depth 3), while Random Forest uses deep unpruned trees?
3. **Missing Values & Scaling**: Why are decision trees completely invariant to monotonic feature scaling (e.g., taking the logarithm or standardizing features)?

---

## Personal Notes
*(Use this space to record your thoughts, notes, and reflections.)*
