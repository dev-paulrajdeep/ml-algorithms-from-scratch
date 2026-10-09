# Chapter 5: Dimensionality Reduction

## Chapter Objectives
- Understand projection from high-dimensional spaces to lower-dimensional spaces.
- Learn how Principal Component Analysis (PCA) preserves maximum dataset variance.
- Learn how Linear Discriminant Analysis (LDA) maximizes class separability in supervised settings.
- Compare and contrast supervised vs. unsupervised dimensionality reduction.

## Prerequisites
- **Stage 0 Links**:
  - [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) (eigenvalues, eigenvectors, covariance matrices)
  - [Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics) (sample variance and covariance)
- **Recommended Context**: Chapter 4 (Clustering) — PCA is highly useful for visualizing clusters in 2D/3D.

## Recommended Study Sequence
1. [PCA (Principal Component Analysis)](pca.py)
2. [LDA (Linear Discriminant Analysis)](lda.py)

## Chapter 5 Toy Datasets
These small datasets will help verify your implementations manually:
- **Dataset 5A (Perfect correlation)**: `X = [[1, 2], [2, 4], [3, 6], [4, 8]]`
  - The first principal component captures 100% of the variance. The second eigenvalue is exactly 0.
- **Dataset 5B (2D rotated ellipse)**: Data generated along a diagonal axis.
  - Captures variance along the diagonal axis.
- **Dataset 5C (Supervised overlap)**: A scenario where PCA fails to separate classes but LDA succeeds.
  - PCA projection onto the maximum variance axis merges classes.
  - LDA finds the projection onto the maximum separation axis, cleanly dividing the classes.

## Granular Progression
- **Core Operations**: Feature scaling, data centering ($X - \mu$), variance calculation, and linear projection $z = X v$.
- **PCA Standardization**: Understand that PCA standardization is context-dependent. Investigate when it is appropriate (different feature units) vs. when raw variance carries signal.
- **LDA Scatter Matrices**: Distinguish and compute within-class scatter $S_W$ vs. between-class scatter $S_B$.

## Exercise Index
- [`pca.py`](pca.py): Principal Component Analysis via eigendecomposition of the covariance matrix. Covers mean-centering, context-dependent standardization, variance explained, and data reconstruction.
- [`lda.py`](lda.py): Linear Discriminant Analysis. A supervised method optimizing the ratio of between-class scatter to within-class scatter, limited to `(n_classes - 1)` components.

## Cross-Chapter Conceptual Questions
1. If you run a classification algorithm from Chapter 2 on PCA-reduced data, what might be the pros and cons compared to using LDA?
2. When does dimensionality reduction help machine learning models? When might it hurt their performance?
3. How does the objective of PCA (variance preservation) fundamentally differ from the objective of LDA (class separability)?

## Chapter Math Learning Goals
- Understand the computation of covariance matrices and their properties.
- Grasp eigendecomposition (eigenvalues and eigenvectors) as it applies to real symmetric matrices.
- Learn how to compute within-class and between-class scatter matrices.

## Completion Checklist
- [ ] Implement PCA and successfully reconstruct data.
- [ ] Understand when to standardize data before PCA and when not to.
- [ ] Implement LDA and limit components correctly.
- [ ] Understand why LDA is limited to `n_classes - 1` components.

## Personal Notes
*(Leave your notes and reflections here)*
