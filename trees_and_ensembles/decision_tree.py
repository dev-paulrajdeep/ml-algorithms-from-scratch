r"""
A. **Mission** (Step 0: Understand the Problem)
The Decision Tree (specifically CART: Classification and Regression Trees) is a non-parametric model that recursively partitions the input space into axis-aligned rectangular boxes, fitting a simple constant prediction (class mode or numerical mean) in each terminal region.
Real-world problem: Medical triage rules, loan approval logic, or customer churn diagnosis where transparent, human-interpretable "if-this-then-that" decision rules are required.
Unlike linear models, decision trees can capture highly non-linear feature interactions without feature transformations, but unconstrained trees will greedily memorize training data and overfit.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Boolean masking, unique values, sorting, and array slicing.
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) — Probability distributions, sample variance, and Shannon entropy.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Overfitting, pruning, and model complexity.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why does an unpruned decision tree with `max_depth = None` always achieve $100\%$ training accuracy on distinct training samples? Why does this indicate severe overfitting?
2. Why are decision tree boundaries strictly axis-aligned (perpendicular to feature axes)? How does a tree approximate a smooth diagonal boundary like $x_1 = x_2$?
3. How does the splitting criterion differ between classification (maximizing information gain / purity) and regression (maximizing variance reduction)?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider Dataset T1: $X = [[1.0], [2.0], [5.0], [6.0]]$, $y = [0, 0, 1, 1]$.
   - Parent node: 2 of class 0, 2 of class 1 ($p_0 = 0.5, p_1 = 0.5$).
     Parent Gini: $G_{\text{parent}} = 1 - (0.5^2 + 0.5^2) = 1 - 0.5 = 0.5$.
   - Test candidate threshold $t = 3.5$ on feature 0:
     Left subset ($x \le 3.5$): $y_L = [0, 0]$ (2 samples). $G_{\text{left}} = 1 - (1.0^2 + 0) = 0.0$.
     Right subset ($x > 3.5$): $y_R = [1, 1]$ (2 samples). $G_{\text{right}} = 1 - (0 + 1.0^2) = 0.0$.
   - Weighted child impurity: $\frac{2}{4}(0.0) + \frac{2}{4}(0.0) = 0.0$.
   - Information Gain (Impurity Decrease): $\Delta G = 0.5 - 0.0 = 0.5$ (Maximum possible gain!).
2. **Step 3: Mathematical Notation**:
   - $N$: number of samples at current node, $D$: number of features.
   - $k \in \{0, \dots, K-1\}$: class labels; $p_k$: proportion of class $k$ samples in node.
   - Feature index $j$, threshold $s$.
   - Left partition: $D_L = \{i : x_{ij} \le s\}$; Right partition: $D_R = \{i : x_{ij} > s\}$.
3. **Step 4: Derive the Impurity Criteria**:
   - **Gini Impurity** (Classification):
     $$G(D) = 1 - \sum_{k=1}^K p_k^2$$
   - **Entropy** (Classification):
     $$H(D) = -\sum_{k=1}^K p_k \log_2(p_k) \quad (\text{with } 0 \log 0 = 0)$$
   - **Variance Reduction** (Regression):
     $$\text{Var}(D) = \frac{1}{|D|} \sum_{i \in D} (y_i - \bar{y})^2$$
   - **Information Gain / Impurity Decrease**:
     $$\Delta I(D, j, s) = I(D) - \left( \frac{|D_L|}{|D|} I(D_L) + \frac{|D_R|}{|D|} I(D_R) \right)$$
     The algorithm greedily searches for $(j^*, s^*) = \arg\max_{j, s} \Delta I(D, j, s)$.

E. **Implementation Contract**
- Class: `DecisionTree`
- Constructor:
  - `__init__(self, task: str = 'classification', criterion: str = 'gini', max_depth: int = None, min_samples_split: int = 2)`
- Methods:
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Builds the tree recursively starting from root.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Traverses the fitted tree for each query vector and returns predicted class mode (classification) or mean (regression). Returns shape `(N,)`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
class Node:
  feature_idx, threshold, left_child, right_child, value

build_tree(X, y, depth):
  If depth == max_depth or len(y) < min_samples_split or pure(y):
    return Node(value = mode(y) if classification else mean(y))

  best_gain = -1
  best_feature, best_threshold = None, None
  For each feature j in range(D):
    For each threshold s in midpoints(unique(X[:, j])):
      gain = compute_information_gain(y, X[:, j], s)
      if gain > best_gain:
        best_gain, best_feature, best_threshold = gain, j, s

  if best_gain <= 0:
    return Node(value = mode(y) if classification else mean(y))

  left_mask = X[:, best_feature] <= best_threshold
  left = build_tree(X[left_mask], y[left_mask], depth + 1)
  right = build_tree(X[~left_mask], y[~left_mask], depth + 1)
  return Node(best_feature, best_threshold, left, right)
```

**Checkpoint 1: Impurity Functions**
- **What to learn**: Quantifying node label mixing.
- **What to do**: Implement `_gini(y)`, `_entropy(y)`, and `_variance(y)`.
- **How to check yourself**: An array of identical labels `[1, 1, 1]` must yield Gini $= 0.0$, Entropy $= 0.0$, Variance $= 0.0$. An evenly balanced binary array `[0, 1]` yields Gini $= 0.5$, Entropy $= 1.0$.
- **When to proceed**: Impurity functions pass all test arrays.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Compute class proportions using `np.bincount(y) / len(y)`.
  - *Hint 2 (Operation)*: In entropy, handle zero probabilities with `p[p > 0]`.
  - *Hint 3 (Debugging)*: For regression, `np.var(y)` computes variance directly.

**Checkpoint 2: Best Split Search**
- **What to learn**: Exhaustive evaluation of split candidates.
- **What to do**: Loop over all features and unique thresholds, compute impurity decrease, and return the maximizing pair $(j, s)$.
- **How to check yourself**: On Dataset T1, the search must identify feature 0 with threshold $\approx 3.5$.
- **When to proceed**: Finds optimal split on 1D and 2D toy datasets.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Good candidate thresholds are midpoints between sorted unique values: `(values[:-1] + values[1:]) / 2`.
  - *Hint 2 (Operation)*: If all values of a feature are identical, skip that feature.
  - *Hint 3 (Debugging)*: Ensure weighted sum of child impurities matches subset sizes.

**Checkpoint 3: Recursive Tree Construction & Prediction**
- **What to learn**: Recursion termination and tree traversal.
- **What to do**: Implement recursive `_build_tree` and recursive prediction `_predict_one(node, x)`.
- **How to check yourself**: On Dataset T2, an unconstrained tree fits 100% of samples. Setting `max_depth = 1` stops at a single split stump.
- **When to proceed**: `predict(X)` returns valid predictions of shape `(N,)`.
- **Recovery hints**:
  - *Hint 1 (Concept)*: A leaf node stores `value` and has `left_child = None, right_child = None`.
  - *Hint 2 (Operation)*: At leaf, return `value`. Otherwise, recurse left if `x[node.feature] <= node.threshold` else right.
  - *Hint 3 (Debugging)*: Check recursion base case: if `len(np.unique(y)) == 1`, return a leaf immediately!

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_pure_dataset`: A dataset where all samples share the same label must terminate immediately as a single root leaf node of depth 0.
2. `test_max_depth_enforcement`: Ensure the tree depth never exceeds `max_depth`. A stump (`max_depth = 1`) contains at most 2 leaves.
3. `test_identical_features_different_labels`: When two samples have identical feature coordinates but conflicting labels, the tree cannot split further and must predict the majority class.
4. `test_regression_mode`: Test continuous targets $y$; verify that variance reduction creates step-like piecewise constant approximations.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Finding best split at node: $O(D \cdot N \log N)$ (sorting features) or $O(D \cdot N)$ with presorting.
  - Full tree construction: $O(D \cdot N \log N \cdot \text{depth})$. In worst case (degenerate tree), $O(D \cdot N^2)$.
  - Prediction per sample: $O(\text{depth}) \approx O(\log N)$ for balanced trees.
- **Space Complexity**:
  - Memory: $O(\text{number of nodes})$ where max nodes $\le 2N$.
- **Trade-offs**: Fast inference, interpretable, invariant to monotonic scaling, but high variance and prone to overfitting without depth limits.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement a depth-1 decision stump for classification using Gini on Dataset T1.
- **Level 2 (Standard)**: Complete recursive `DecisionTree` class supporting both classification (Gini) and regression (variance reduction) with configurable `max_depth` and `min_samples_split`.
- **Level 3 (Challenge)**: Add Entropy criterion, handle multi-class labels, and implement post-pruning based on minimal cost-complexity.
- **Definition of Done**: Correctly splits on Dataset T1, enforces `max_depth`, supports classification and regression, and all unit tests pass.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does a small change in the training dataset often produce a completely different tree structure at the root?
2. How does limiting `max_depth` act as regularization, and how does it balance bias vs. variance?
"""
