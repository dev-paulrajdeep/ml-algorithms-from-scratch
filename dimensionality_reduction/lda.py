"""
A. **Mission**
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction algorithm. Unlike PCA, which maximizes variance blindly, LDA aims to maximize the separability between known classes. It projects the data onto a lower-dimensional space where instances of the same class are close together, and instances of different classes are far apart.

B. **Prerequisites**
- Hard: Covariance, matrix inversion, eigenvalues/eigenvectors, Bayes' theorem basics.
- Helpful: Understanding PCA.

C. **Learning Questions**
1. What is the fundamental difference in the objective function of PCA versus LDA?
2. Why is the maximum number of components in LDA strictly limited to (n_classes - 1)?
3. How does LDA balance "between-class scatter" (separability) and "within-class scatter" (compactness)?

D. **Mathematics to Derive**
1. Formulate the within-class scatter matrix S_W.
2. Formulate the between-class scatter matrix S_B.
3. Write the generalized eigenvalue problem that LDA solves: S_B w = lambda S_W w. How does this translate to finding the eigenvectors of (S_W^-1 S_B)?

E. **Implementation Contract**
- Class: `LDA`
- Arguments: `n_components` (int, optional, default None). If None, defaults to min(n_classes - 1, n_features).
- State: `scalings_` (shape: [n_features, n_components]), `means_` (class means)
- Methods:
  - `fit(X, y)`: Computes the scatter matrices and their eigendecomposition. X is [N, D], y is [N].
  - `transform(X)`: Projects X into the new space. Returns shape [N, n_components].
- NumPy is allowed.

F. **Guided Implementation Stages**
1. **Class Statistics**: Compute the overall mean of X, and the mean vector for each distinct class in y.
2. **Scatter Matrices**:
   - Compute `S_W` (within-class scatter) by summing the covariance matrices of each class (scaled by class sample size).
   - Compute `S_B` (between-class scatter) by measuring the outer product of the difference between each class mean and the overall mean, weighted by class size.
3. **Eigendecomposition**: Compute the eigenvalues and eigenvectors of `(S_W^-1) @ S_B`. (Note: use `np.linalg.inv` or `np.linalg.pinv` for S_W, then `np.linalg.eig`).
4. **Sorting**: Sort the eigenvalues (take their real parts, as numerical issues might produce tiny complex parts) and eigenvectors in descending order.
5. **Selection**: Select the top `n_components` eigenvectors. Store as `scalings_`. Ensure `n_components` <= n_classes - 1.
6. **Projection (`transform`)**: Project X by taking the dot product of X and `scalings_`.

G. **Edge Cases and Expected Tests**
1. Test with `n_components` > `n_classes - 1`. The algorithm should gracefully cap the components at `n_classes - 1`.
2. Test on a linearly separable 2-class dataset. The 1D projection should perfectly separate the classes.
3. Test that eigenvectors and eigenvalues contain only real numbers (drop imaginary parts if they are near zero).

H. **Complexity Analysis**
- Time Complexity: What is the bottleneck step? Is it computing S_W, or taking its inverse?
- Space Complexity: How does LDA's memory requirement scale with the number of classes and features?

I. **Definition of Done**
- `fit` and `transform` methods are implemented.
- The `n_components` limit of `n_classes - 1` is strictly enforced.
- The algorithm handles singular S_W gracefully (e.g., via pseudoinverse).

J. **Reflection**
- Under what data distributions might LDA fail to find a good separation?
- Can LDA be used as a classifier directly, or is it purely for dimensionality reduction?
"""
