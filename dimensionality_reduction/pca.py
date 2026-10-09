"""
A. **Mission**
Principal Component Analysis (PCA) is an unsupervised dimensionality reduction algorithm. It projects high-dimensional data onto a lower-dimensional subspace while preserving as much variance as possible. It is useful for data visualization, noise reduction, and mitigating the curse of dimensionality.

B. **Prerequisites**
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) (eigenvalues, eigenvectors, covariance matrices)
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) (sample variance and covariance)

C. **Learning Questions**
1. Why must the data be mean-centered before performing PCA?
2. Standardization (scaling features to unit variance) before PCA is context-dependent. Under what circumstances is standardization appropriate? When might it be unnecessary or even destructive to the signal?
3. How do the eigenvalues correspond to the amount of variance explained by their respective principal components?
4. What happens if you reduce the data to D dimensions (where D is the original number of features) and then reconstruct it?

D. **Mathematics to Derive**
1. Let X be an N x D mean-centered dataset. Derive the D x D covariance matrix formula in matrix notation.
2. Formulate the relationship between the covariance matrix, its eigenvectors, and its eigenvalues (Rayleigh quotient connection).
3. Formulate the projection of X onto the top K eigenvectors to produce the reduced dataset.
4. Formulate the reconstruction of the original dataset from the reduced dataset.

E. **Implementation Contract**
- Class: `PCA`
- Arguments: `n_components` (int, optional, default None). If None, keep all components.
- State: `components_` (shape: [n_components, n_features]), `explained_variance_ratio_` (shape: [n_components]), `mean_` (shape: [n_features])
- Methods:
  - `fit(X)`: Computes the principal components and explained variance. X is shape [N, D].
  - `transform(X)`: Projects X into the lower-dimensional space. Returns shape [N, n_components].
  - `inverse_transform(X_reduced)`: Projects the reduced data back to the original space. Returns shape [N, D].
- NumPy is allowed for array operations and `np.linalg.eigh` or `np.linalg.svd`.

F. **Guided Implementation Stages**
**Checkpoint 1: Mean Centering**
- What to learn: Center data without losing relative distances.
- What to do: Compute the feature-wise mean of X and subtract it from X. Store this mean.
- How to check yourself: `np.mean(centered_X, axis=0)` should be near zero.
- When to proceed: Centered means are essentially zero.
- Recovery hints: 1) Ensure you specify `axis=0` for mean. 2) Check broadcasting. 3) Retain the mean for reconstruction.

**Checkpoint 2: Covariance Matrix & Eigendecomposition**
- What to learn: Extract variance directions.
- What to do: Compute the covariance matrix of the mean-centered data. Find its eigenvalues and eigenvectors using `np.linalg.eigh`.
- How to check yourself: Covariance matrix should be symmetric.
- When to proceed: Eigenvalues are all non-negative.
- Recovery hints: 1) Covariance is `(X.T @ X) / (N - 1)`. 2) `eigh` is for symmetric matrices. 3) Order of eigenvalues might be ascending.

**Checkpoint 3: Sorting Components & Selection**
- What to learn: Prioritize directions with most variance.
- What to do: Sort the eigenvalues in descending order, and sort the eigenvectors accordingly. Select the top `n_components`.
- How to check yourself: First selected eigenvalue is the largest.
- When to proceed: Components matrix has shape `[n_components, n_features]`.
- Recovery hints: 1) `np.argsort` gives ascending order; reverse it. 2) Ensure eigenvectors are sorted corresponding to eigenvalues. 3) Store eigenvectors as rows in `components_`.

**Checkpoint 4: Explained Variance**
- What to learn: Quantify information retained.
- What to do: Compute the `explained_variance_ratio_` by dividing the selected eigenvalues by the sum of all eigenvalues.
- How to check yourself: Ratios should be positive and sum to <= 1.0.
- When to proceed: Calculation yields a valid probability-like array.
- Recovery hints: 1) Sum all original eigenvalues for the denominator. 2) Ratio sum is exactly 1 if all components are kept.

**Checkpoint 5: Projection (`transform`) & Reconstruction (`inverse_transform`)**
- What to learn: Move back and forth between spaces.
- What to do: In `transform`, mean-center X using `mean_`, then dot product with `components_.T`. In `inverse_transform`, multiply reduced data by `components_` and add back `mean_`.
- How to check yourself: Reconstructed data matches original if `n_components == n_features`.
- When to proceed: Shapes match expected outputs.
- Recovery hints: 1) Remember to center data in transform. 2) Inverse transform does not re-center, it adds the mean back. 3) Watch out for matrix shapes in dot product.

G. **Edge Cases and Expected Tests**
1. Test with zero-variance features: Eigenvalue for that feature should be exactly zero.
2. Test rank-deficient covariance (e.g., N < D or highly correlated features like Dataset 5A).
3. Test that applying `fit` to data with vastly different scales heavily biases the first principal component, illustrating when to use context-dependent standardization.

H. **Complexity Analysis**
- Time Complexity: What is the complexity of computing the covariance matrix and its eigendecomposition?
- Space Complexity: How much memory is required to store the components and the covariance matrix?

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Implement with provided NumPy functions step-by-step.
- Level 2 Standard: Handle edge cases and implement robust testing.
- Level 3 Challenge: Implement PCA using Singular Value Decomposition (SVD) directly on X instead of the covariance matrix.
- Definition of Done: All methods work, variance ratios sum correctly, and reconstruction is accurate.

J. **Reflection**
- How does the context-dependency of standardization in PCA challenge the idea of an "always correct" preprocessing pipeline?
- Why might we use explained variance ratio rather than a hard-coded number of components to choose `n_components`?
"""
