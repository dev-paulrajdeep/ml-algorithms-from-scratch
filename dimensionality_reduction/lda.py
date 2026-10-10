r"""
A. **Mission** (Step 0: Understand the Problem)
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction and classification technique.
Real-world problem: Face recognition (Fisherfaces), medical biomarker selection, or speech processing where we seek a lower-dimensional projection that maximizes class separation.
Unlike PCA (which blindly seeks directions of maximum total variance without knowing class labels), LDA finds an optimal linear combination of features that **maximizes between-class separability while minimizing within-class variance**.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Boolean masking per class, outer products, and matrix inversion (`np.linalg.pinv`).
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Eigendecomposition of non-symmetric matrices (`np.linalg.eig`), matrix rank, and scatter matrices.
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) — Class-conditional means and covariance matrices.
- [Chapter 2: Classification](../classification/README.md) — Class labels and categorical centroids.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why does PCA fail to separate the classes in Dataset P3 while LDA cleanly separates them?
2. Why is the maximum number of components in LDA strictly bounded by $C - 1$ (where $C$ is the number of distinct classes)?
3. What is the geometric meaning of the "between-class scatter" matrix $\mathbf{S}_B$ vs. the "within-class scatter" matrix $\mathbf{S}_W$?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider Dataset P3 with 2 classes in 2D:
   Class 0: $\mathbf{x}_1 = [0, -1], \mathbf{x}_2 = [0, 1] \implies \boldsymbol{\mu}_0 = [0, 0], N_0 = 2$.
   Class 1: $\mathbf{x}_3 = [2, -1], \mathbf{x}_4 = [2, 1] \implies \boldsymbol{\mu}_1 = [2, 0], N_1 = 2$.
   - Overall mean: $\boldsymbol{\mu} = [1, 0]$.
   - Total variance is largest along the vertical axis (from $-1$ to $+1$), so PCA projects onto the vertical axis, merging both classes!
   - In contrast, class centroids differ along the horizontal axis ($\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 = [2, 0]$).
   - Within-class scatter $\mathbf{S}_W$:
     Class 0: $([0, -1] - [0, 0])^T ([0, -1] - [0, 0]) + ([0, 1] - [0, 0])^T ([0, 1] - [0, 0]) = \begin{bmatrix} 0 & 0 \\ 0 & 2 \end{bmatrix}$.
     Class 1: Same scatter = $\begin{bmatrix} 0 & 0 \\ 0 & 2 \end{bmatrix} \implies \mathbf{S}_W = \begin{bmatrix} 0 & 0 \\ 0 & 4 \end{bmatrix}$.
   - Between-class scatter $\mathbf{S}_B$:
     $N_0 (\boldsymbol{\mu}_0 - \boldsymbol{\mu})(\boldsymbol{\mu}_0 - \boldsymbol{\mu})^T + N_1 (\boldsymbol{\mu}_1 - \boldsymbol{\mu})(\boldsymbol{\mu}_1 - \boldsymbol{\mu})^T = 2 \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} + 2 \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 4 & 0 \\ 0 & 0 \end{bmatrix}$.
   - The optimal projection direction is along the horizontal axis $[1, 0]^T$, achieving perfect separation!
2. **Step 3: Mathematical Notation**:
   - $N$: total samples, $D$: features, $C$: number of classes.
   - $\boldsymbol{\mu}_c \in \mathbb{R}^D$: mean vector of class $c$; $\boldsymbol{\mu} \in \mathbb{R}^D$: overall dataset mean.
   - $\mathbf{S}_W \in \mathbb{R}^{D \times D}$: Within-Class Scatter Matrix.
   - $\mathbf{S}_B \in \mathbb{R}^{D \times D}$: Between-Class Scatter Matrix.
   - $\mathbf{w} \in \mathbb{R}^D$: projection direction.
3. **Step 4: Derive the Fisher Criterion and Generalized Eigenvalue Problem**:
   - Define the Fisher Criterion (Rayleigh quotient):
     $$J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$
   - **Within-Class Scatter**:
     $$\mathbf{S}_W = \sum_{c=1}^C \sum_{\mathbf{x} \in \text{Class } c} (\mathbf{x} - \boldsymbol{\mu}_c)(\mathbf{x} - \boldsymbol{\mu}_c)^T$$
   - **Between-Class Scatter**:
     $$\mathbf{S}_B = \sum_{c=1}^C N_c (\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T$$
   - To maximize $J(\mathbf{w})$, take the derivative with respect to $\mathbf{w}$ and set it to $\mathbf{0}$:
     $$\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$$
   - Multiplying by $\mathbf{S}_W^{-1}$ yields the standard eigenvalue problem:
     $$\mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w}$$
   - *The $C - 1$ Rank Proof*: Since $\mathbf{S}_B$ is the sum of $C$ rank-1 outer products of vectors $(\boldsymbol{\mu}_c - \boldsymbol{\mu})$, and these vectors sum to zero ($\sum N_c (\boldsymbol{\mu}_c - \boldsymbol{\mu}) = \mathbf{0}$), $\mathbf{S}_B$ has rank at most $C - 1$. Therefore, there are at most $C - 1$ non-zero eigenvalues!

E. **Implementation Contract**
- Class: `LDA`
- Constructor:
  - `__init__(self, n_components: int = None)`:
    Stores requested component count.
- Attributes:
  - `scalings_`: array of shape `(D, n_components)` containing discriminant projection vectors as columns.
  - `means_`: array of shape `(C, D)` containing class centroids.
  - `classes_`: array of unique class labels.
- Methods:
  - `fit(self, X: np.ndarray, y: np.ndarray) -> self`:
    Computes $\mathbf{S}_W$ and $\mathbf{S}_B$, solves the generalized eigenvalue problem, caps components at $\min(n\_components, C - 1, D)$, and stores `scalings_`.
  - `transform(self, X: np.ndarray) -> np.ndarray`:
    Projects $X$ onto `scalings_`: `X @ self.scalings_`. Returns shape `(N, n_components)`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X, y):
  classes = unique(y)
  C = len(classes)
  N, D = X.shape
  overall_mean = mean(X, axis=0)
  
  S_W = zeros((D, D))
  S_B = zeros((D, D))
  For c in classes:
    X_c = X[y == c]
    mean_c = mean(X_c, axis=0)
    diff = X_c - mean_c
    S_W += diff.T @ diff
    
    mean_diff = (mean_c - overall_mean).reshape(-1, 1)
    S_B += len(X_c) * (mean_diff @ mean_diff.T)
    
  # Regularize S_W slightly for numerical stability: S_W += 1e-4 * I
  matrix_to_solve = pinv(S_W) @ S_B
  eigenvalues, eigenvectors = eig(matrix_to_solve)
  
  # Take real parts and sort descending
  eigenvalues = real(eigenvalues)
  eigenvectors = real(eigenvectors)
  idx = argsort(eigenvalues)[::-1]
  
  max_allowed = min(C - 1, D)
  k = min(n_components, max_allowed) if n_components else max_allowed
  scalings_ = eigenvectors[:, idx[:k]]
```

**Checkpoint 1: Class Means and Within-Class Scatter $\mathbf{S}_W$**
- **What to learn**: Summing class-conditional covariance matrices.
- **What to do**: Loop over unique classes, compute each class mean, and accumulate `diff.T @ diff` into `S_W`.
- **How to check yourself**: $\mathbf{S}_W$ is symmetric, shape `(D, D)`, and its diagonal entries are non-negative.
- **When to proceed**: $\mathbf{S}_W$ matches manual scatter computation on isolated classes.
- **Recovery hints**:
  - *Hint 1 (Concept)*: $\mathbf{S}_W$ measures total dispersion of points around their own class centers.
  - *Hint 2 (Operation)*: `diff = X[y == c] - mean_c; S_W += diff.T @ diff`.
  - *Hint 3 (Debugging)*: If $N < D$, $\mathbf{S}_W$ is singular; add small jitter `1e-4 * np.eye(D)`.

**Checkpoint 2: Between-Class Scatter $\mathbf{S}_B$**
- **What to learn**: Computing dispersion between class centroids.
- **What to do**: For each class, compute outer product of $(\boldsymbol{\mu}_c - \boldsymbol{\mu})$ scaled by class size $N_c$.
- **How to check yourself**: $\mathbf{S}_B$ has shape `(D, D)` and rank at most $C - 1$.
- **When to proceed**: On 2-class data, $\mathbf{S}_B$ has rank exactly 1.
- **Recovery hints**:
  - *Hint 1 (Concept)*: $\mathbf{S}_B$ is a weighted sum of rank-1 matrices.
  - *Hint 2 (Operation)*: Use `mean_diff = (mean_c - overall_mean).reshape(-1, 1); S_B += len(X_c) * (mean_diff @ mean_diff.T)`.
  - *Hint 3 (Debugging)*: Check `assert S_B.shape == (D, D)`.

**Checkpoint 3: Eigendecomposition & Component Capping**
- **What to learn**: Solving $\mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w}$ and enforcing the $C - 1$ dimension cap.
- **What to do**: Compute eigenvalues/vectors using `np.linalg.eig`, extract `.real`, sort descending, and cap components.
- **How to check yourself**: For $C=2$ classes, output component count is strictly 1 regardless of feature dimension $D$.
- **When to proceed**: 1D projection of Dataset P3 cleanly separates the two classes without overlap.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Because $\mathbf{S}_W^{-1} \mathbf{S}_B$ is generally not symmetric, use `np.linalg.eig` (not `eigh`).
  - *Hint 2 (Operation)*: Use `np.real()` to discard tiny imaginary artifacts ($< 10^{-15}i$) caused by numerical precision.
  - *Hint 3 (Debugging)*: Enforce `k = min(n_components, C - 1)`.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_binary_classification_max_components`: On any 2-class dataset, LDA must cap components to at most $C - 1 = 1$, regardless of whether the user requests `n_components = 5`.
2. `test_supervised_separation_dataset_p3`: On Dataset P3 (where PCA collapses classes), 1D LDA projection must achieve 100% linear separability between Class 0 and Class 1.
3. `test_singular_within_class_scatter`: Test on a dataset where $N < D$ (more features than samples). Ensure pseudoinverse `np.linalg.pinv` or diagonal jitter prevents a singular matrix crash.
4. `test_shape_consistency`: `transform(X)` returns shape `(N, n_components)`.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Scatter matrix computation: $O(N \cdot D^2)$ time.
  - Matrix inverse and product: $O(D^3)$ time.
  - Eigendecomposition: $O(D^3)$ time.
  - Total fitting: $O(N D^2 + D^3)$.
  - Projection: $O(N_{\text{test}} \cdot D \cdot k)$.
- **Space Complexity**: $O(D^2)$ auxiliary memory for scatter matrices $\mathbf{S}_W$ and $\mathbf{S}_B$.
- **Trade-offs**: Optimal for linear class separation under Gaussian assumptions, but limited to at most $C - 1$ dimensions and sensitive to outliers.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement 2-class, 2-feature LDA on Dataset P3 by calculating scatter matrices and finding the 1D projection vector.
- **Level 2 (Standard)**: Complete `LDA` class supporting multi-class data, automatic component capping at $C - 1$, pseudoinverse handling, and projection.
- **Level 3 (Challenge)**: Extend LDA to act as a **Classifier** directly (assigning query samples to the nearest class centroid in projected space using Mahalanobis distance).
- **Definition of Done**: Correctly projects Dataset P3 to separate classes, enforces the $C - 1$ component bound, and handles singular scatter matrices gracefully.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does LDA assume that all classes share the exact same covariance matrix? (Hint: What happens when classes have unequal covariances $\boldsymbol{\Sigma}_1 \neq \boldsymbol{\Sigma}_2$, leading to Quadratic Discriminant Analysis?).
2. Under what conditions would an unsupervised PCA projection outperform a supervised LDA projection when training a downstream classifier?
"""
