"""
A. **Mission**
Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction algorithm. Unlike PCA, which maximizes variance blindly, LDA aims to maximize the separability between known classes. It projects the data onto a lower-dimensional space where instances of the same class are close together, and instances of different classes are far apart.

B. **Prerequisites**
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) (eigenvalues, eigenvectors, covariance matrices)
- [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) (sample variance and covariance)

C. **Learning Questions**
1. What is the fundamental difference in the objective function of PCA versus LDA?
2. Why is the maximum number of components in LDA strictly limited to (n_classes - 1)?
3. How does LDA balance "between-class scatter" (separability) and "within-class scatter" (compactness)?

D. **Mathematics to Derive**
1. Formulate the within-class scatter matrix S_W.
2. Formulate the between-class scatter matrix S_B.
3. Write the generalized eigenvalue problem that LDA solves: S_B w = lambda S_W w. How does this translate to finding the eigenvectors of (S_W^-1 S_B)? Show the objective function for LDA.

E. **Implementation Contract**
- Class: `LDA`
- Arguments: `n_components` (int, optional, default None). If None, defaults to min(n_classes - 1, n_features).
- State: `scalings_` (shape: [n_features, n_components]), `means_` (class means)
- Methods:
  - `fit(X, y)`: Computes the scatter matrices and their eigendecomposition. X is [N, D], y is [N].
  - `transform(X)`: Projects X into the new space. Returns shape [N, n_components].
- NumPy is allowed.

F. **Guided Implementation Stages**
**Checkpoint 1: Class Statistics**
- What to learn: Extract class-specific properties.
- What to do: Compute the overall mean of X, and the mean vector for each distinct class in y.
- How to check yourself: Overall mean should equal the weighted average of class means.
- When to proceed: Class means are stored correctly.
- Recovery hints: 1) Ensure you slice X using y correctly. 2) Compute means across axis 0. 3) Track unique classes.

**Checkpoint 2: Scatter Matrices**
- What to learn: Measure within-class and between-class variance.
- What to do: Compute `S_W` (within-class scatter) by summing the covariance matrices of each class (scaled by class sample size). Compute `S_B` (between-class scatter) by measuring the outer product of the difference between each class mean and the overall mean, weighted by class size.
- How to check yourself: S_W and S_B should both be shape [D, D].
- When to proceed: Both matrices are symmetric and positive semi-definite.
- Recovery hints: 1) `S_W` is the sum of scatter matrices for each class `(X_c - mean_c).T @ (X_c - mean_c)`. 2) `S_B` uses `N_c * (mean_c - mean_overall).reshape(-1, 1) @ ...`. 3) Verify dimensions before adding.

**Checkpoint 3: Eigendecomposition**
- What to learn: Solve the generalized eigenvalue problem.
- What to do: Compute the eigenvalues and eigenvectors of `(S_W^-1) @ S_B`.
- How to check yourself: `np.linalg.eig` works but might yield complex numbers due to numerical instability.
- When to proceed: Eigenvalues and vectors are computed.
- Recovery hints: 1) Use `np.linalg.pinv` for S_W if it's singular. 2) Real parts can be extracted using `.real` if tiny imaginary components appear.

**Checkpoint 4: Sorting and Selection**
- What to learn: Select the best discriminant axes.
- What to do: Sort the eigenvalues in descending order and match eigenvectors. Select top `n_components` eigenvectors as `scalings_`. Ensure `n_components` <= `n_classes - 1`.
- How to check yourself: Capping logic handles large requested `n_components`.
- When to proceed: `scalings_` matrix is formulated.
- Recovery hints: 1) Double check sorting logic. 2) Columns of the eigenvector matrix are the directions. 3) Enforce the `n_classes - 1` limit explicitly.

**Checkpoint 5: Projection (`transform`)**
- What to learn: Apply the learned transformation.
- What to do: Project X by taking the dot product of X and `scalings_`.
- How to check yourself: Shapes match. Dataset 5C perfectly separates.
- When to proceed: Transformation yields robust lower-dimensional features.
- Recovery hints: 1) Unlike PCA, LDA doesn't strictly need mean-centering before projection since we care about relative separability. 2) Just compute `X @ scalings_`.

G. **Edge Cases and Expected Tests**
1. Test with `n_components` > `n_classes - 1`. The algorithm should gracefully cap the components at `n_classes - 1`.
2. Test on a linearly separable 2-class dataset. The 1D projection should perfectly separate the classes.
3. Test rank-deficient covariance/zero-variance features.

H. **Complexity Analysis**
- Time Complexity: What is the bottleneck step? Is it computing S_W, or taking its inverse?
- Space Complexity: How does LDA's memory requirement scale with the number of classes and features?

I. **Progressive Difficulty Levels & Definition of Done**
- Level 1 Guided: Implement core scatter matrices and eigendecomposition.
- Level 2 Standard: Handle singular S_W using pseudoinverse and enforce component limits.
- Level 3 Challenge: Formulate LDA as a multi-class extension of Fisher's Linear Discriminant without relying directly on `eig` of the product.
- Definition of Done: `fit` and `transform` are implemented, limit of `n_classes - 1` is strictly enforced, and singular matrices are handled gracefully.

J. **Reflection**
- Under what data distributions might LDA fail to find a good separation?
- Can LDA be used as a classifier directly, or is it purely for dimensionality reduction?
"""
