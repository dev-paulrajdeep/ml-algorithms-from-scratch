r"""
A. **Mission** (Step 0: Understand the Problem)
K-Nearest Neighbors (KNN) is a non-parametric, instance-based "lazy learning" algorithm used for classification and regression.
Real-world problem: Medical diagnosis based on patient symptom profiles or customer segmentation based on purchase behavior. Instead of learning an explicit mathematical equation during training, KNN defers all computation until inference: to classify an unseen patient, it finds the $K$ most similar historical patients in the database and assigns the majority diagnosis.
Your objective is to implement a robust KNN classifier from scratch, handle distance calculations, implement clean tie-breaking, and analyze the computational cost of lazy evaluation.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Array broadcasting, `np.argsort`, and `np.bincount`.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Euclidean distance and $L_2$ norm ($\|\mathbf{u} - \mathbf{v}\|_2$).
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Bias-variance tradeoff and feature scaling.

C. **Learning Questions** (Step 1: Build Intuition)
1. How does the choice of $K$ govern model complexity? What happens when $K=1$ (very low bias, high variance, memorization) vs. $K=N$ (predicts the global majority class)?
2. Why is KNN called a "lazy learner"? What are the consequences for memory consumption and inference latency in production?
3. What is the "curse of dimensionality", and why do Euclidean distances between points become nearly equidistant as the number of features $D$ grows large?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider 3 training points in 1D: $X_{\text{train}} = [[1.0], [2.0], [5.0]]$, $y_{\text{train}} = [0, 0, 1]$.
   Given query point $x_{\text{test}} = [3.0]$ and $K = 2$:
   - Hand-compute distances:
     $d(3, 1) = |3 - 1| = 2.0$
     $d(3, 2) = |3 - 2| = 1.0$
     $d(3, 5) = |3 - 5| = 2.0$
   - Identify the 2 smallest distances: $1.0$ (index 1, label 0) and $2.0$ (index 0 or 2).
   - If breaking ties by smaller index: indices $[1, 0]$ have labels $[0, 0]$, majority vote predicts class $0$.
2. **Step 3: Mathematical Notation**:
   - $N$: number of training samples, $D$: number of features.
   - $\mathbf{X}_{\text{train}} \in \mathbb{R}^{N \times D}$, $\mathbf{y}_{\text{train}} \in \{0, 1, \dots, C-1\}^N$.
   - $\mathbf{x} \in \mathbb{R}^D$: test sample.
   - $d(\mathbf{x}, \mathbf{x}_i) = \|\mathbf{x} - \mathbf{x}_i\|_2 = \sqrt{\sum_{j=1}^D (x_j - x_{ij})^2}$: Euclidean distance.
   - $\mathcal{N}_K(\mathbf{x})$: set of indices of the $K$ training samples with smallest distance to $\mathbf{x}$.
3. **Step 4: Derive the Majority Vote Rule**:
   - The predicted label $\hat{y}$ is the mode (most frequent label) among the neighbors:
     $$\hat{y} = \arg\max_{c \in \{0, \dots, C-1\}} \sum_{i \in \mathcal{N}_K(\mathbf{x})} \mathbb{I}(y_i = c)$$
   - Define a deterministic tie-breaking policy: when two classes have equal vote counts, break ties by selecting the class with the smaller average distance, or the class appearing first.

E. **Implementation Contract**
- Class: `KNearestNeighbors`
- Methods:
  - `__init__(self, k: int = 3)`:
    Stores hyperparameter $k$. Validates that $k \ge 1$.
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Stores training features `X_train_` of shape `(N, D)` and labels `y_train_` of shape `(N,)`. No computation performed.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Computes distances from each test sample to all stored training samples, selects $k$ nearest, and predicts labels. Returns 1D array of shape `(M,)` where $M = X.shape[0]$.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X, y):
  store X_train_ = copy(X)
  store y_train_ = copy(y)

predict_single(x):
  distances = sqrt(sum((X_train_ - x)**2, axis=1))
  nearest_indices = argsort(distances)[:k]
  nearest_labels = y_train_[nearest_indices]
  return mode(nearest_labels)

predict(X):
  return array([predict_single(x) for x in X])
```

**Checkpoint 1: Storing Data in `fit`**
- **What to learn**: The lazy learning paradigm.
- **What to do**: Store `self.X_train_` and `self.y_train_` in `fit`. Verify types and shapes.
- **How to check yourself**: Ensure attributes exist and `self.X_train_.shape == X.shape`.
- **When to proceed**: Immediately after assigning instance variables.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Training in KNN requires zero floating-point operations.
  - *Hint 2 (Operation)*: Use `np.asarray(X)` to guarantee NumPy array format.
  - *Hint 3 (Debugging)*: Check `assert len(X) == len(y)`.

**Checkpoint 2: Vectorized Distance Calculation**
- **What to learn**: Computing Euclidean distances from one test sample to all $N$ training rows.
- **What to do**: Use NumPy broadcasting: `diff = self.X_train_ - x_test; dists = np.sqrt(np.sum(diff**2, axis=1))`.
- **How to check yourself**: The distance from any training point to itself must be $0.0$.
- **When to proceed**: Distances match hand calculation on tiny 3-point examples.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Broadcasting subtracts 1D vector `(D,)` from 2D matrix `(N, D)`.
  - *Hint 2 (Operation)*: Ensure `axis=1` in `sum`.
  - *Hint 3 (Debugging)*: You can use `np.linalg.norm(self.X_train_ - x_test, axis=1)`.

**Checkpoint 3: Selecting Neighbors and Voting**
- **What to learn**: Extracting top $K$ smallest indices and computing the statistical mode.
- **What to do**: Use `np.argsort(dists)[:self.k]` to get neighbor indices, retrieve their labels, and use `np.bincount` to find the most frequent label.
- **How to check yourself**: If nearest labels are `[1, 1, 0]`, mode is `1`.
- **When to proceed**: Correctly classifies a single query point on Dataset C1.
- **Recovery hints**:
  - *Hint 1 (Concept)*: `np.argsort` returns indices sorted in ascending numerical order.
  - *Hint 2 (Operation)*: `counts = np.bincount(nearest_labels); return np.argmax(counts)`.
  - *Hint 3 (Debugging)*: For multi-class labels, ensure `nearest_labels` contains non-negative integers.

**Checkpoint 4: Full Batch Inference**
- **What to learn**: Generalizing to a matrix of $M$ test samples.
- **What to do**: Loop over all rows of $X$ or vectorize the full pairwise distance matrix.
- **How to check yourself**: Output shape is exactly `(M,)`.
- **When to proceed**: Passes basic accuracy tests on Dataset C1.
- **Recovery hints**:
  - *Hint 1 (Concept)*: A list comprehension `[self._predict_one(x) for x in X]` is the simplest Level 1 approach.
  - *Hint 2 (Operation)*: Convert list of predictions to `np.array(preds)`.
  - *Hint 3 (Debugging)*: Check `assert len(preds) == len(X)`.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_k_equals_1_memorization`: When $K=1$, evaluating on the training set must achieve exactly $100\%$ accuracy (0 training error).
2. `test_deterministic_tie_breaking`: When $K=2$ and neighbor labels are $[0, 1]$, the algorithm must break ties consistently without crashing or raising errors.
3. `test_feature_scaling_sensitivity`: Create a dataset where feature 1 is in range $[0, 1]$ and feature 2 is in range $[0, 10000]$. Show that unscaled data misclassifies, while standardized data classifies correctly.
4. `test_k_equals_n`: When $K=N$, predictions for all test points must equal the global majority class of the training set.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Training: $O(1)$ time (simply stores references).
  - Inference per query: $O(N \cdot D + N \log K)$ time (distance computation + partial sort).
  - Full test set of size $M$: $O(M \cdot N \cdot D)$.
- **Space Complexity**:
  - Storage: $O(N \cdot D)$ auxiliary memory to store all training vectors.
- **Trade-offs**: Fast training, but heavy inference and memory footprint. Unsuitable for real-time mobile inference on millions of rows.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement distance calculation and majority voting using a loop over test points.
- **Level 2 (Standard)**: Complete `KNearestNeighbors` class supporting multi-class labels, tie-breaking policy, and validation.
- **Level 3 (Challenge)**: Fully vectorize the pairwise distance computation across all $M$ test points simultaneously using matrix algebra:
  $$\|\mathbf{X}_{\text{test}} - \mathbf{X}_{\text{train}}\|^2 = \sum \mathbf{X}_{\text{test}}^2 - 2 \mathbf{X}_{\text{test}} \mathbf{X}_{\text{train}}^T + \sum \mathbf{X}_{\text{train}}^2$$
  eliminating all Python loops over samples!
- **Definition of Done**: All tests pass, $K=1$ memorizes training data, and handles multi-class ties deterministically.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. How does the decision boundary of KNN become smoother as $K$ increases from $1$ to $15$?
2. Why is KNN completely incapable of extrapolating patterns outside the convex hull of the training data?
"""
