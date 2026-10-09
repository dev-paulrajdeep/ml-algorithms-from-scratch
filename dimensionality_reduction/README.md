# Chapter 5: Dimensionality Reduction

## Chapter Objectives
- Understand projection from high-dimensional spaces to lower-dimensional spaces.
- Learn how Principal Component Analysis (PCA) preserves maximum dataset variance.
- Learn how Linear Discriminant Analysis (LDA) maximizes class separability in supervised settings.
- Compare and contrast supervised vs. unsupervised dimensionality reduction.

## Prerequisites
- **Hard Prerequisites**: Linear algebra (eigenvalues, eigenvectors, covariance matrices, general matrix operations), and statistical variance concepts (Chapter 1).
- **Recommended Context**: Chapter 4 (Clustering) — PCA is highly useful for visualizing clusters in 2D/3D, though clustering does not explicitly require PCA.

## Recommended Study Sequence
1. [PCA (Principal Component Analysis)](pca.py)
2. [LDA (Linear Discriminant Analysis)](lda.py)

## Exercise Index
- [`pca.py`](pca.py): Principal Component Analysis via eigendecomposition of the covariance matrix. Covers mean-centering, context-dependent standardization, variance explained, and data reconstruction.
- [`lda.py`](lda.py): Linear Discriminant Analysis. A supervised method optimizing the ratio of between-class scatter to within-class scatter, limited to `(n_classes - 1)` components.

## Cross-Chapter Conceptual Questions
1. If you run a classification algorithm from Chapter 2 on PCA-reduced data, what might be the pros and cons compared to using LDA?
2. When does dimensionality reduction help machine learning models? When might it hurt their performance?
3. How does the objective of PCA (variance preservation) fundamentally differ from the objective of LDA (class separability)?

## Algorithm Relationships
- **PCA vs. LDA**: Both involve eigendecomposition, but PCA acts on the covariance matrix (unsupervised) to capture directions of maximum variance. LDA acts on scatter matrices (supervised) to find directions that best separate labeled classes. While PCA's components are limited by the number of features, LDA's components are fundamentally bottlenecked by `n_classes - 1`.

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
