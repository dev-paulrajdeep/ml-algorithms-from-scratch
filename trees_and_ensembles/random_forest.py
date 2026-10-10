r"""
A. **Mission** (Step 0: Understand the Problem)
Random Forest is a parallel ensemble learning method that constructs a multitude of decision trees at training time and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.
Real-world problem: High-variance tabular datasets (e.g., credit default risk, medical diagnosis, Kaggle competitions) where individual trees severely overfit noise.
By combining **Bootstrap Aggregating (Bagging)** with **Random Feature Subsampling**, Random Forest decorrelates the individual trees, dramatically slashing variance without increasing bias.

B. **Prerequisites**
- [Decision Tree](./decision_tree.py) — Must understand single tree behavior, node splitting, and leaf predictions first.
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — `np.random.choice` with replacement.
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) — Sampling with replacement, variance of sums of random variables.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — The bias-variance tradeoff.

C. **Learning Questions** (Step 1: Build Intuition)
1. If all individual trees in an ensemble are identical, does averaging them reduce variance? Why is **decorrelation** between trees the mathematical secret to ensemble performance?
2. What fraction of unique training samples is expected to be included in a bootstrap sample of size $N$ as $N \to \infty$? What happens to the remaining $\approx 36.8\%$ of samples (Out-of-Bag)?
3. Why does Random Forest subsample features at every split (e.g., evaluating only $\sqrt{D}$ features) rather than giving every tree access to all features?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider a dataset of 5 samples: $S = \{A, B, C, D, E\}$.
   - Draw a bootstrap sample with replacement: e.g., $\{A, A, C, D, E\}$ (size is 5; $B$ was omitted).
   - If 3 separate trees predict on query point $x$: Tree 1 predicts $1$, Tree 2 predicts $0$, Tree 3 predicts $1$.
   - Classification vote: Mode of $\{1, 0, 1\}$ is class $1$.
   - Regression average: Mean of $\{1, 0, 1\}$ is $\frac{2}{3} \approx 0.667$.
2. **Step 3: Mathematical Notation**:
   - $M$: number of trees (`n_trees`).
   - $N$: number of training samples, $D$: total number of features.
   - $m_{\text{try}}$: number of features subsampled at each split (typically $\lfloor\sqrt{D}\rfloor$ for classification, $\lfloor D/3 \rfloor$ for regression).
   - $T_m(\mathbf{x})$: prediction of the $m$-th individual decision tree.
3. **Step 4: Derive the Variance Reduction Formula**:
   - Suppose we average $M$ identically distributed (but correlated) random variables, each with variance $\sigma^2$ and pairwise correlation $\rho \in [0, 1]$:
     $$\text{Var}\left( \frac{1}{M} \sum_{m=1}^M T_m(\mathbf{x}) \right) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$
   - Notice the two terms:
     - As $M \to \infty$, the second term $\frac{1 - \rho}{M} \sigma^2 \to 0$.
     - The irreducible floor is $\rho \sigma^2$. To minimize total variance, we must minimize correlation $\rho$ between trees! Random feature subsampling accomplishes this.
   - **Out-of-Bag (OOB) Limit**:
     The probability that a specific sample is NOT chosen in $N$ draws with replacement is $\left(1 - \frac{1}{N}\right)^N$.
     As $N \to \infty$, $\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = e^{-1} \approx 0.3679$.
     Thus, approximately $36.8\%$ of samples are omitted per tree and can serve as a free validation set!

E. **Implementation Contract**
- Class: `RandomForest`
- Constructor:
  - `__init__(self, n_trees: int = 10, max_depth: int = None, min_samples_split: int = 2, max_features: str = 'sqrt', task: str = 'classification')`
- Methods:
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Generates $M$ bootstrap samples, fits $M$ internal `DecisionTree` instances, and stores them in `trees_`.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Collects predictions from all $M$ trees and computes majority vote (classification) or mean (regression). Returns shape `(N,)`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X, y):
  trees_ = []
  N, D = X.shape
  For m in range(n_trees):
    # Bootstrap sampling
    boot_indices = random_choice(N, size=N, replace=True)
    X_boot, y_boot = X[boot_indices], y[boot_indices]
    
    # Train decision tree with max_features constraint
    tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, task=task)
    tree.fit(X_boot, y_boot)
    trees_.append(tree)

predict(X):
  all_preds = array([tree.predict(X) for tree in trees_]) # shape (n_trees, N_test)
  If task == 'classification':
    return mode(all_preds, axis=0)
  Else:
    return mean(all_preds, axis=0)
```

**Checkpoint 1: Bootstrap Sampling Generator**
- **What to learn**: Generating valid bootstrap samples using NumPy.
- **What to do**: Implement `_bootstrap_sample(X, y)` using `np.random.choice(N, size=N, replace=True)`.
- **How to check yourself**: Verify that the sample has length $N$, contains duplicates, and omits approximately $30-40\%$ of original row indices.
- **When to proceed**: The bootstrap function correctly indexes rows of both $X$ and $y$ simultaneously.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Sampling with replacement allows the same row index to appear multiple times.
  - *Hint 2 (Operation)*: `indices = np.random.choice(len(X), size=len(X), replace=True); return X[indices], y[indices]`.
  - *Hint 3 (Debugging)*: Never sample $X$ and $y$ independently—they must share the exact same indices!

**Checkpoint 2: Training Multiple Trees**
- **What to learn**: Ensemble construction using base learners.
- **What to do**: Loop `n_trees` times, instantiate and fit a `DecisionTree` on each bootstrap sample, and append to `self.trees_`.
- **How to check yourself**: Ensure `len(self.trees_) == n_trees` and each tree has distinct node splits.
- **When to proceed**: All $M$ trees fit successfully without errors.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Each tree in the ensemble is an independent estimator.
  - *Hint 2 (Operation)*: Store fitted tree instances in a standard Python list.
  - *Hint 3 (Debugging)*: If individual trees do not support `max_features`, randomly select a subset of feature columns for each tree as a Level 1 approximation.

**Checkpoint 3: Aggregating Predictions**
- **What to learn**: Aggregation mechanisms (voting vs. averaging).
- **What to do**: Collect predictions of shape `(n_trees, N_test)` and reduce along `axis=0`.
- **How to check yourself**: For classification, output labels are valid class integers; for regression, output is continuous.
- **When to proceed**: Model passes accuracy tests on Dataset T1 and Dataset T2.
- **Recovery hints**:
  - *Hint 1 (Concept)*: For classification, use `scipy.stats.mode` or `np.apply_along_axis(lambda col: np.bincount(col).argmax(), axis=0, arr=all_preds)`.
  - *Hint 2 (Operation)*: For regression, `np.mean(all_preds, axis=0)`.
  - *Hint 3 (Debugging)*: Check `assert preds.shape == (X.shape[0],)`.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_single_tree_matches_decision_tree`: When `n_trees = 1`, Random Forest behavior closely mirrors an individual decision tree.
2. `test_variance_reduction_on_noisy_curve`: Train on Dataset T3 (noisy sine wave). Show that Random Forest with $M=50$ produces a much smoother prediction curve with lower test MSE than a single unpruned tree.
3. `test_reproducibility_with_seed`: Setting a fixed random seed produces identical predictions across multiple runs.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Training: $O(M \cdot N \log N \cdot m_{\text{try}} \cdot \text{depth})$. Since all $M$ trees are completely independent, training is embarrassingly parallelizable ($O(1/P)$ speedup with $P$ CPU cores).
  - Inference: $O(M \cdot \text{depth})$ per sample.
- **Space Complexity**: $O(M \cdot \text{nodes per tree})$. Memory consumption scales linearly with $M$.
- **Trade-offs**: Dramatically reduces overfitting, but sacrifices the single-tree interpretability (we cannot easily visualize 100 trees).

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement bootstrap sampling and ensemble voting using 5 Decision Trees on Dataset T1.
- **Level 2 (Standard)**: Complete `RandomForest` class supporting configurable `n_trees`, `max_depth`, classification mode, and regression mode.
- **Level 3 (Challenge)**: Implement **Out-of-Bag (OOB) error estimation** (evaluating each sample only on the subset of trees where it was omitted during bootstrapping) to evaluate performance without a separate validation split.
- **Definition of Done**: Successfully reduces variance on noisy data, implements proper bootstrap aggregation, and passes all edge tests.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does adding more trees ($M = 100 \to 500$) never cause a Random Forest to overfit, but yields diminishing returns in accuracy?
2. How does the feature subsampling parameter $m_{\text{try}}$ balance the correlation $\rho$ between trees against the strength (individual accuracy) of each tree?
"""
