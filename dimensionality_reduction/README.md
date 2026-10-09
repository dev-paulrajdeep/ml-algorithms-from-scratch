# Chapter 5: Dimensionality Reduction

> **Hard Prerequisites:** Linear algebra (dot products, matrix multiplication, eigenvalues/eigenvectors) and the concept of variance (Chapter 1).
>
> **Recommended context:** Chapter 4 (PCA is often used to visualise clusters, but clustering does not require PCA).

---

## Learning Objectives

After completing this chapter, you should be able to:

- Explain how high-dimensional data can be projected into fewer dimensions while preserving meaningful structure
- Distinguish unsupervised reduction (PCA) from supervised reduction (LDA)
- Implement eigendecomposition-based dimensionality reduction from scratch
- Reason about when dimensionality reduction helps and when it hurts

## Questions to Answer in Your Own Words

1. What does it mean to "explain variance"? What does a principal component represent geometrically?
2. When is standardisation (zero mean, unit variance) appropriate before PCA, and when might it not be necessary or even desirable?
3. What is an eigenvalue and eigenvector, and what role do they play in PCA?
4. How does LDA differ from PCA in its objective?
5. When would you choose LDA over PCA, and vice versa?
6. How do you decide how many components to keep?

## Algorithms to Implement from Scratch

- [ ] Principal Component Analysis (PCA)
- [ ] Linear Discriminant Analysis (LDA)

## Mathematical Derivations to Complete

- Derive the covariance matrix from first principles
- Show that PCA eigenvectors of the covariance matrix maximise projected variance
- Derive the LDA objective: maximise the ratio of between-class scatter to within-class scatter

## Edge Cases & Tests to Consider

- What happens when two features are perfectly correlated? (Eigenvalue = 0)
- Reconstruct original data from reduced components and measure reconstruction error
- Verify that principal components are orthogonal (dot product = 0)
- Apply PCA to data where features have very different scales, with and without standardisation — compare results and explain

## Completion Criteria

You are done with this chapter when you can:

- [ ] Run PCA on a dataset and plot the explained variance ratio
- [ ] Choose the number of components using the explained variance curve
- [ ] Reconstruct data from k < d components and quantify information loss
- [ ] Explain the difference between PCA and LDA without notes
- [ ] Apply PCA as a visualisation tool for clusters from Chapter 4

## Notes

_Space for your own observations as you work through this chapter._
