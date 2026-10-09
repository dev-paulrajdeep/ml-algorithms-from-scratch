# Chapter 4: Clustering

## Chapter Objectives
In this chapter, we explore **unsupervised learning**, specifically finding structure without labels through cluster discovery. We aim to group similar data points together and evaluate performance using intrinsic evaluation metrics rather than external ground truth.

## Prerequisites
- **Stage 0 Links**:
  - [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) (norms and distance metrics).
  - [Stage 0F: ML Core](../foundations/README.md#module-0f-machine-learning-core-concepts) (unsupervised framing).

*Note: Clustering is taught independently of dimensionality reduction. PCA is NOT a prerequisite for clustering. While PCA can optionally be used to visualize high-dimensional clusters, it is not required.*

## Recommended Study Sequence
For the best conceptual progression, tackle the algorithms in this order:
1. **K-Means**: Centroid-based clustering.
2. **DBSCAN**: Density-based clustering.
3. **Agglomerative Clustering**: Hierarchical clustering.

## Chapter 4 Toy Datasets
Use these simple synthetic datasets to verify behavior during implementation:
- **Dataset 4A (Two 2D clusters with clear gap)**: Cluster 1 around (1, 1), Cluster 2 around (5, 5). Ideal for basic K-Means sanity checks.
- **Dataset 4B (Non-convex concentric rings / two moons)**: K-Means fails here; DBSCAN succeeds in tracking continuous density regions.
- **Dataset 4C (Varying density clusters)**: Challenges DBSCAN with a single epsilon parameter; useful for comparing linkage behaviors in Agglomerative clustering.

## Granular Progression
- **Start with manual grouping**: Given 4 coordinates on a graph, manually assign them to 2 centroids by calculating distances, and then recompute the cluster means by hand.
- **Understand the paradigms**: Contrast the Centroid vs. Density vs. Hierarchical clustering paradigms.

## Exercise Index
- [`kmeans.py`](./kmeans.py): Implement Lloyd's algorithm, tracking WCSS (inertia), with K-Means++ initialization.
- [`dbscan.py`](./dbscan.py): Implement epsilon-neighborhoods and core/border/noise classification.
- [`agglomerative_clustering.py`](./agglomerative_clustering.py): Build a hierarchical dendrogram comparing single, complete, and average linkage.

## Cross-Chapter Conceptual Questions
1. **When to use what?** When is K-Means appropriate versus DBSCAN? When would you prefer Agglomerative Clustering?
2. **Centroid vs. Density vs. Hierarchical**: 
   - How does minimizing variance to a centroid (K-Means) differ fundamentally from tracing continuous dense regions (DBSCAN)?
   - How does bottom-up merging contrast with iterative refinement?
3. **Evaluation without Labels**: How can we intrinsically argue that one set of clusters is "better" than another?

## Chapter Math Learning Goals
- Prove that K-Means iterations (assignment + update) never increase the Within-Cluster Sum of Squares (WCSS) objective.
- Formulate the $\epsilon$-neighborhood and density reachability mathematically.
- Derive the linkage formulas (single, complete, average) from pairwise point distances.

## Completion Checklist
- [ ] Grouped a 4-point toy dataset manually.
- [ ] Completed `kmeans.py` (Level 3 challenge passed).
- [ ] Completed `dbscan.py` (Level 3 challenge passed).
- [ ] Completed `agglomerative_clustering.py` (Level 3 challenge passed).

## Personal Notes
> Use this space to record your insights, derivation notes, or "aha!" moments.
