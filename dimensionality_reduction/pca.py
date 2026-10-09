"""
A. **Mission**
Principal Component Analysis (PCA) is an unsupervised dimensionality reduction algorithm. It projects high-dimensional data onto a lower-dimensional subspace while preserving as much variance as possible. It is useful for data visualization, noise reduction, and mitigating the curse of dimensionality.

B. **Prerequisites**
- Hard: Matrix multiplication, covariance calculation, eigendecomposition (eigenvalues, eigenvectors).
- Helpful: Geometric intuition of projection and basis vectors.

C. **Learning Questions**
1. Why must the data be mean-centered before performing PCA?
2. Standardization (scaling features to unit variance) before PCA is context-dependent. Under what circumstances is standardization appropriate? When might it be unnecessary or even destructive to the signal?
3. How do the eigenvalues correspond to the amount of variance explained by their respective principal components?
4. What happens if you reduce the data to D dimensions (where D is the original number of features) and then reconstruct it?

D. **Mathematics to Derive**
1. Let X be an N x D mean-centered dataset. Derive the D x D covariance matrix formula in matrix notation.
2. Formulate the relationship between the covariance matrix, its eigenvectors, and its eigenvalues.
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
1. **Mean Centering**: Compute the feature-wise mean of X and subtract it from X. Store this mean for `inverse_transform`.
2. **Covariance Matrix**: Compute the covariance matrix of the mean-centered data.
3. **Eigendecomposition**: Find the eigenvalues and eigenvectors of the covariance matrix. (Hint: `np.linalg.eigh` is preferred for symmetric matrices).
4. **Sorting Components**: Sort the eigenvalues in descending order, and sort the eigenvectors accordingly.
5. **Selection**: Select the top `n_components` eigenvectors. Store them as `components_` (typically as rows).
6. **Explained Variance**: Compute the `explained_variance_ratio_` by dividing the selected eigenvalues by the sum of all eigenvalues.
7. **Projection (`transform`)**: Mean-center the input X using the fitted `mean_`, then take the dot product with the transpose of `components_`.
8. **Reconstruction (`inverse_transform`)**: Multiply the reduced data by the `components_` matrix and add back the `mean_`.

G. **Edge Cases and Expected Tests**
1. Test that `inverse_transform(transform(X))` perfectly reconstructs X when `n_components` equals the number of features.
2. Test that `explained_variance_ratio_` sums to exactly 1.0 when `n_components` equals the number of features.
3. Test that applying `fit` to data with vastly different scales (e.g., [1, 1000]) heavily biases the first principal component toward the feature with the large scale, proving the need for mindful standardization.

H. **Complexity Analysis**
- Time Complexity: What is the complexity of computing the covariance matrix and its eigendecomposition for an N x D dataset?
- Space Complexity: How much memory is required to store the components and the covariance matrix?

I. **Definition of Done**
- `fit`, `transform`, and `inverse_transform` are implemented.
- `transform` correctly projects data.
- Reconstruction error is near zero when keeping all components.
- Standardization vs. raw scale behavior is manually tested and understood.

J. **Reflection**
- How does the context-dependency of standardization in PCA challenge the idea of an "always correct" preprocessing pipeline?
- Why might we use explained variance ratio rather than a hard-coded number of components to choose `n_components`?
"""
