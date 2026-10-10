r"""
A. **Mission** (Step 0: Understand the Problem)
Principal Component Analysis (PCA) is the primary unsupervised linear dimensionality reduction algorithm. It projects high-dimensional data onto a lower-dimensional orthogonal subspace that maximizes the retained variance of the data, which mathematically minimizes the squared reconstruction error.
Real-world problem: Visualizing 100-dimensional gene expressions or customer profiles in 2D/3D scatter plots, compressing high-resolution images, and removing collinearity before training linear models.
Your objective is to implement PCA from scratch using covariance eigendecomposition, handle mean-centering, compute explained variance ratios, and reconstruct data via inverse transformations.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Mean-centering with broadcasting, matrix multiplication (`@`), and `np.linalg.eigh`.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Eigenvalues, eigenvectors of symmetric matrices, dot product projections, and orthogonal bases.
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) — Sample variance, covariance, and the covariance matrix.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why is **mean-centering** strictly mandatory before calculating the covariance matrix in PCA? What happens to the principal components if data is not centered at the origin?
2. Under what conditions is **feature standardization** ($z$-score scaling to unit variance) necessary before PCA, and under what conditions is it destructive to true variance signals?
3. How do the eigenvalues $\lambda_j$ correspond to the amount of variance explained by their respective principal component eigenvectors?
4. What happens if you reduce $D$-dimensional data to $D$ components (retaining all components) and then call `inverse_transform`?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider Dataset P1 (perfect correlation): $X = [[1, 2], [2, 4], [3, 6], [4, 8]]$ ($N=4, D=2$).
   - Mean vector: $\bar{\mathbf{x}} = [2.5, 5.0]$.
   - Centered matrix $\mathbf{X}_c = [[-1.5, -3.0], [-0.5, -1.0], [0.5, 1.0], [1.5, 3.0]]$.
   - Note that column 2 is exactly $2 \times$ column 1!
   - Compute covariance $\mathbf{C} = \frac{1}{3} \mathbf{X}_c^T \mathbf{X}_c$:
     $\sum x_{c1}^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5.0 \implies C_{11} = 5/3$.
     $\sum x_{c1} x_{c2} = 4.5 + 0.5 + 0.5 + 4.5 = 10.0 \implies C_{12} = 10/3$.
     $\sum x_{c2}^2 = 9.0 + 1.0 + 1.0 + 9.0 = 20.0 \implies C_{22} = 20/3$.
     $\mathbf{C} = \frac{5}{3} \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$.
   - Determinant of $\mathbf{C}$ is $(1)(4) - (2)(2) = 0.0$.
   - Eigenvalues: $\lambda_1 = \text{Tr}(\mathbf{C}) = \frac{5}{3}(5) = \frac{25}{3} \approx 8.333$; $\lambda_2 = 0.0$.
   - The first principal component explains $\frac{8.333}{8.333 + 0} = 100\%$ of the variance!
2. **Step 3: Mathematical Notation**:
   - $N$: number of samples, $D$: original feature dimensions, $k$: reduced component count ($k \le D$).
   - $\mathbf{X} \in \mathbb{R}^{N \times D}$, $\bar{\mathbf{x}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i \in \mathbb{R}^D$.
   - $\mathbf{X}_c = \mathbf{X} - \mathbf{1} \bar{\mathbf{x}}^T \in \mathbb{R}^{N \times D}$: mean-centered matrix.
   - $\mathbf{C} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c \in \mathbb{R}^{D \times D}$: sample covariance matrix.
   - $\mathbf{v}_j \in \mathbb{R}^D$: orthonormal eigenvector of $\mathbf{C}$, with corresponding eigenvalue $\lambda_j$.
   - $\mathbf{W}_k = [\mathbf{v}_1, \dots, \mathbf{v}_k] \in \mathbb{R}^{D \times k}$: projection matrix containing top $k$ eigenvectors as columns.
3. **Step 4: Derive the Projection, Reconstruction, and Variance Ratios**:
   - **Forward Projection** (`transform`):
     $$\mathbf{Z} = \mathbf{X}_c \mathbf{W}_k \in \mathbb{R}^{N \times k}$$
   - **Inverse Reconstruction** (`inverse_transform`):
     $$\hat{\mathbf{X}} = \mathbf{Z} \mathbf{W}_k^T + \mathbf{1} \bar{\mathbf{x}}^T \in \mathbb{R}^{N \times D}$$
   - **Explained Variance Ratio**:
     $$\text{EVR}_j = \frac{\lambda_j}{\sum_{m=1}^D \lambda_m}$$

E. **Implementation Contract**
- Class: `PCA`
- Constructor:
  - `__init__(self, n_components: int = None)`:
    Stores $k$. If `None`, retains all $D$ components.
- Attributes:
  - `components_`: array of shape `(n_components, D)` containing principal direction vectors as rows.
  - `explained_variance_`: array of shape `(n_components,)` with eigenvalues.
  - `explained_variance_ratio_`: array of shape `(n_components,)` with normalized ratios in $[0, 1]$.
  - `mean_`: array of shape `(D,)` with original feature column means.
- Methods:
  - `fit(self, X: np.ndarray) -> self`:
    Computes `mean_`, centered covariance, eigendecomposition, sorts components descending, and stores attributes.
  - `transform(self, X: np.ndarray) -> np.ndarray`:
    Centers $X$ using `self.mean_` and projects onto `self.components_.T`. Returns shape `(N, n_components)`.
  - `fit_transform(self, X: np.ndarray) -> np.ndarray`:
    Calls `fit(X)` followed by `transform(X)`.
  - `inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray`:
    Projects reduced coordinates back to original space: `X_reduced @ self.components_ + self.mean_`. Returns shape `(N, D)`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X):
  mean_ = mean(X, axis=0)
  X_c = X - mean_
  N, D = X.shape
  cov_matrix = (X_c.T @ X_c) / (N - 1)
  eigenvalues, eigenvectors = eigh(cov_matrix)
  
  # eigh returns ascending order; sort descending
  idx = argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:, idx]
  
  k = n_components if n_components is not None else D
  components_ = eigenvectors[:, :k].T # shape (k, D)
  explained_variance_ = eigenvalues[:k]
  explained_variance_ratio_ = eigenvalues[:k] / sum(eigenvalues)

transform(X):
  X_c = X - mean_
  return X_c @ components_.T # shape (N, k)

inverse_transform(Z):
  return Z @ components_ + mean_ # shape (N, D)
```

**Checkpoint 1: Mean Centering & Covariance Matrix**
- **What to learn**: Safe computation of sample covariance.
- **What to do**: Store `self.mean_ = np.mean(X, axis=0)`. Compute `cov = (X_c.T @ X_c) / (N - 1)`.
- **How to check yourself**: `cov` must be a symmetric square matrix of shape `(D, D)`. Its diagonal values match `np.var(X, axis=0, ddof=1)`.
- **When to proceed**: Centered matrix has mean $\approx 0.0$ and covariance is symmetric.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Unbiased sample covariance divides by $N - 1$, not $N$.
  - *Hint 2 (Operation)*: Verify symmetry: `assert np.allclose(cov, cov.T)`.
  - *Hint 3 (Debugging)*: Never modify $X$ in-place; compute `X_c = X - self.mean_`.

**Checkpoint 2: Eigendecomposition and Sorting**
- **What to learn**: Extracting principal directions using `np.linalg.eigh`.
- **What to do**: Call `eigenvalues, eigenvectors = np.linalg.eigh(cov)`. Reverse them to sort in descending order of variance.
- **How to check yourself**: The first eigenvalue is the largest. On Dataset P1, $\lambda_1 > 0$ and $\lambda_2 \approx 0.0$.
- **When to proceed**: Eigenvalues are non-negative real numbers sorted in strictly descending order.
- **Recovery hints**:
  - *Hint 1 (Concept)*: `np.linalg.eigh` is specifically optimized for real symmetric matrices.
  - *Hint 2 (Operation)*: `np.linalg.eigh` returns ascending order; use `[::-1]` to reverse.
  - *Hint 3 (Debugging)*: Remember that eigenvectors are the **columns** of `eigenvectors[:, idx]`.

**Checkpoint 3: Projection & Lossless Reconstruction**
- **What to learn**: Linear transformation to lower-dimensional space and reconstruction.
- **What to do**: Implement `transform` and `inverse_transform`.
- **How to check yourself**: If `n_components == D`, `inverse_transform(transform(X))` must equal `X` within numerical floating-point precision ($10^{-12}$).
- **When to proceed**: Reconstruction is lossless when keeping all components.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Orthogonal matrices satisfy $\mathbf{W}^T \mathbf{W} = \mathbf{I}$.
  - *Hint 2 (Operation)*: In `inverse_transform`, remember to add back `self.mean_`.
  - *Hint 3 (Debugging)*: Check matrix shapes: `(N, k) @ (k, D) + (D,)` yields `(N, D)`.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_lossless_reconstruction_all_components`: Projecting to $k=D$ and calling `inverse_transform` reconstructs the original dataset with error $< 10^{-10}$.
2. `test_perfect_correlation_dataset_p1`: On Dataset P1, verify that `explained_variance_ratio_[0]` is exactly $1.0$ (within floating tolerance) and `explained_variance_ratio_[1]` is $0.0$.
3. `test_zero_variance_feature`: When a feature is constant across all samples, its corresponding eigenvalue is $0.0$.
4. `test_shape_consistency`: For input shape `(N, D)`, output of `transform` must be `(N, k)` and output of `inverse_transform` must be `(N, D)`.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Covariance matrix computation: $O(N \cdot D^2)$ time.
  - Eigendecomposition of $(D, D)$ matrix: $O(D^3)$ time.
  - Total fitting: $O(N D^2 + D^3)$.
  - Projection (`transform`): $O(N_{\text{test}} \cdot D \cdot k)$ time.
  - Reconstruction (`inverse_transform`): $O(N_{\text{test}} \cdot k \cdot D)$ time.
- **Space Complexity**:
  - Covariance matrix: $O(D^2)$ auxiliary memory.
  - Components storage: $O(k \cdot D)$.
- **Trade-offs**: Optimal linear compression, but restricted strictly to linear hyperplanes (cannot unroll non-linear manifolds like a Swiss Roll).

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Project 2D Dataset P1 to 1D, verifying that 100% of variance is captured by the first component.
- **Level 2 (Standard)**: Complete `PCA` class with `fit`, `transform`, `fit_transform`, `inverse_transform`, and explained variance ratio calculations.
- **Level 3 (Challenge)**: Implement PCA using Singular Value Decomposition (`np.linalg.svd`) directly on the centered matrix $\mathbf{X}_c$ without explicitly constructing the $D \times D$ covariance matrix (vital for $D > N$).
- **Definition of Done**: Reconstructs data losslessly with all components, correctly calculates variance ratios, and passes all unit tests.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does computing SVD directly on $\mathbf{X}_c$ ($\mathbf{X}_c = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$) avoid the numerical squaring of condition numbers caused by forming $\mathbf{X}_c^T \mathbf{X}_c$?
2. When would you choose to standardize features before PCA, and when would you leave features unstandardized?
"""
