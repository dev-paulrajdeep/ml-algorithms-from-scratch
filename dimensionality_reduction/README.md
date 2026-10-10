# Chapter 5: Dimensionality Reduction

Welcome to Chapter 5! In this chapter, you will learn how to compress high-dimensional feature spaces into lower-dimensional representations while preserving critical statistical structure. We explore both **unsupervised projection** (Principal Component Analysis) and **supervised projection** (Linear Discriminant Analysis).

---

## Chapter Objectives
- Understand linear projection from $\mathbb{R}^D$ to $\mathbb{R}^k$ ($k < D$).
- Understand the geometry of **variance preservation**: why maximizing projected variance minimizes reconstruction error.
- Master data centering ($\mathbf{X} - \bar{\mathbf{x}}$), sample covariance matrices, and eigendecomposition.
- Understand when feature standardization is necessary vs. when raw variance carries genuine signal (**context-dependent standardization**).
- Contrast unsupervised variance maximization (PCA) with supervised class-separation maximization (LDA).
- Formulate within-class scatter $\mathbf{S}_W$ and between-class scatter $\mathbf{S}_B$, and understand why LDA yields at most $C - 1$ components.

---

## Prerequisites
Before tackling this chapter, ensure you have completed:
- **[Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing)** — Array centering, matrix multiplication (`@`), and eigendecomposition (`np.linalg.eigh`, `np.linalg.eig`).
- **[Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra)** — Dot products as projections, symmetric matrices, eigenvectors, and eigenvalues ($\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$).
- **[Stage 0E: Probability & Statistics](../foundations/README.md#module-0e-probability-and-statistics)** — Sample variance, covariance between features, and class-conditional statistics.
- **[Chapter 2: Classification](../classification/README.md)** — Class labels and categorical centroids (needed for LDA).

---

## 🧭 Pedagogical Progression & Intuition Bridges

To master dimensionality reduction intuitively, absorb these foundations:

### 1. Before PCA: The Concept of Information-Preserving Directions
- Imagine data points scattered along a narrow diagonal cigar in 2D space (e.g., height in inches vs. height in centimeters).
- Knowing both features is redundant—almost all the information is captured along the long diagonal axis.
- If you must project the 2D data onto a 1D line:
  - If you project onto the long axis: the spread (variance) of the points is preserved, and points stay distinguishable.
  - If you project onto the short axis: all points collapse into a dense clump, and individual differences are destroyed!
- **Data Centering**: Before computing variance, you must subtract the feature-wise mean ($\mathbf{X} - \bar{\mathbf{x}}$) so that projections are anchored at the origin.
- **Covariance Matrix**: $\mathbf{C} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c$. The diagonal entries are feature variances; off-diagonal entries are feature covariances.
- **Eigenvectors & Eigenvalues**:
  - The eigenvectors of the covariance matrix point in the orthogonal directions of maximum variance.
  - The corresponding eigenvalues $\lambda_j$ measure the exact amount of variance along each eigenvector axis.
  - The **Explained Variance Ratio** for component $j$ is $\frac{\lambda_j}{\sum \lambda_k}$.

### 2. Context-Dependent Standardization in PCA
- Should you standardize features ($z$-score scaling to variance $1.0$) before running PCA?
  - **YES, standardize** when features are measured in different units (e.g., age in years $[18, 70]$ vs. income in dollars $[20000, 200000]$). Otherwise, income's numeric scale will dominate 99.9% of the variance purely by virtue of its unit of measurement!
  - **NO, do not standardize** when all features share the exact same physical unit (e.g., pixel intensities $[0, 255]$ in computer vision or spatial coordinates) and you want features with genuine high variance to carry more weight than noisy background pixels.

### 3. Before LDA: Supervised Class Separation
- PCA is unsupervised: it has no knowledge of class labels. It might project data along an axis of high variance that completely merges two distinct classes!
- Linear Discriminant Analysis (LDA) is **supervised**: it finds a projection vector $\mathbf{w}$ that:
  1. Maximizes the distance between class means (**between-class scatter** $\mathbf{S}_B$).
  2. Minimizes the variance within each class (**within-class scatter** $\mathbf{S}_W$).
- **The $C - 1$ Component Limit**: Because $C$ class centroids span a subspace of dimension at most $C - 1$, the rank of $\mathbf{S}_B$ is at most $C - 1$. Therefore, LDA can project to at most $C - 1$ dimensions (e.g., exactly 1 dimension for binary classification).

---

## Recommended Study Sequence

1. **[`pca.py`](./pca.py)** — Start here. Master mean-centering, covariance matrix eigendecomposition, variance explained, and inverse reconstruction.
2. **[`lda.py`](./lda.py)** — Supervised projection maximizing the Fisher criterion $\frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$, enforcing the $C - 1$ dimension cap.

---

## Chapter 5 Toy Datasets

- **Dataset P1 (Perfect Correlation)**:
  - $\mathbf{X} = [[1, 2], [2, 4], [3, 6], [4, 8]]$
  - Hand check: Feature 1 is exactly $2 \times$ Feature 0. The first principal component captures $100\%$ of the variance ($\lambda_1 > 0, \lambda_2 = 0$).

- **Dataset P2 (Rotated 2D Ellipse)**:
  - Synthetic Gaussian cloud stretched along the line $y = x$.
  - Hand check: The top eigenvector is $[\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}]^T \approx [0.707, 0.707]^T$.

- **Dataset P3 (Where PCA Fails and LDA Succeeds)**:
  - Class 0: Points elongated vertically from $(0, -5)$ to $(0, 5)$.
  - Class 1: Points elongated vertically from $(2, -5)$ to $(2, 5)$.
  - Hand check: The direction of maximum variance is the vertical axis ($Y$). PCA projects onto $Y$, completely merging Class 0 and Class 1! LDA projects onto the horizontal axis ($X$), cleanly separating the two classes.

---

## Completion Checklist
- [ ] Compute a $2 \times 2$ covariance matrix and eigenvalues by hand for Dataset P1.
- [ ] Implement `pca.py` with mean-centering and reconstruction (`inverse_transform`).
- [ ] Understand when to standardize features before PCA and when not to.
- [ ] Implement `lda.py` with scatter matrices $\mathbf{S}_W$ and $\mathbf{S}_B$.
- [ ] Verify that LDA caps components at $C - 1$.
- [ ] Compare PCA vs LDA projections on Dataset P3.

---

## Cross-Chapter Conceptual Questions
1. **PCA Before KNN**: Why does running PCA to reduce 500 features down to 20 features often drastically improve the accuracy and speed of K-Nearest Neighbors (Chapter 2)?
2. **Reconstruction Loss**: If you project $D$-dimensional data onto $D$ components (keeping all components), what is the error between the original data and `inverse_transform`? (Answer: Exactly zero!).
3. **Singular Within-Class Scatter**: What happens to LDA when the number of samples $N$ is less than the number of features $D$ ($N < D$)? How can you regularize $\mathbf{S}_W$ to ensure invertibility?

---

## Personal Notes
*(Use this space to record your thoughts, notes, and reflections.)*
