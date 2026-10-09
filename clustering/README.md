# Chapter 4: Clustering

## Chapter Objectives
In this chapter, we explore **unsupervised learning**, specifically cluster discovery. We aim to group similar data points together without relying on pre-existing labels. You will learn to implement core clustering algorithms and evaluate their performance using intrinsic metrics rather than external ground truth.

## Prerequisites
- **Hard Prerequisites**: Distance metrics, iterative optimization concepts (from Chapter 1).
- **Recommended**: Chapter 2 for a strong understanding of supervised learning contexts, allowing you to contrast them with the unsupervised methods explored here.

*Note: Clustering is taught independently of dimensionality reduction. PCA is NOT a prerequisite for clustering. While PCA can optionally be used to visualize high-dimensional clusters, it is not required to implement or understand the algorithms in this chapter.*

## Recommended Study Sequence
For the best conceptual progression, tackle the algorithms in this order:
1. **K-Means**: The classic centroid-based optimization approach.
2. **DBSCAN**: A shift to density-based clustering, handling noise and non-convex shapes.
3. **Agglomerative Clustering**: A hierarchical bottom-up approach to understand relationships at multiple scales.

## Exercise Index
- [`kmeans.py`](./kmeans.py): Implement Lloyd's algorithm, K-Means++ initialization, and track the Within-Cluster Sum of Squares (WCSS / inertia) objective.
- [`dbscan.py`](./dbscan.py): Implement density-reachability clustering using $\epsilon$-neighborhoods and core/border/noise point classification.
- [`agglomerative_clustering.py`](./agglomerative_clustering.py): Build a bottom-up hierarchical dendrogram using single, complete, and average linkage criteria.

## Cross-Chapter Conceptual Questions
1. **When to use what?** When is K-Means appropriate versus DBSCAN? When would you prefer Agglomerative Clustering over both?
2. **Centroid vs. Density vs. Hierarchical**:
   - How does the objective of minimizing variance to a centroid (K-Means) differ fundamentally from tracing continuous dense regions (DBSCAN)?
   - How does the bottom-up deterministic merging in hierarchical clustering contrast with the random initialization dependence of K-Means?
3. **Evaluation**: Since we don't have labels (like we did in Chapter 2 classification), how can we mathematically argue that one set of clusters is "better" than another?

## Chapter Math Learning Goals
- Prove that K-Means iterations (assignment and update steps) never increase the objective function.
- Understand probabilistic initialization strategies (K-Means++).
- Formulate $\epsilon$-neighborhoods and density reachability mathematically.
- Derive distance formulas for different hierarchical linkage criteria (single, complete, average) based on pairwise distances.

## Completion Checklist
- [ ] Completed `kmeans.py` with both random and K-Means++ initialization.
- [ ] Confirmed K-Means handles empty cluster edge cases and monotonically decreases inertia.
- [ ] Completed `dbscan.py` with accurate core, border, and noise labeling.
- [ ] Confirmed DBSCAN correctly identifies non-convex clusters like "moons".
- [ ] Completed `agglomerative_clustering.py` and generated a valid merge history.
- [ ] Implemented and tested single, complete, and average linkage.

## Personal Notes
> Use this space to record your insights, derivation notes, or "aha!" moments.
