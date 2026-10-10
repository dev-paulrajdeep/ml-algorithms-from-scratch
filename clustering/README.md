# Chapter 4: Clustering

Welcome to Chapter 4! In this chapter, we step into the domain of **unsupervised machine learning**. Unlike regression and classification where models learn from ground-truth target labels, clustering algorithms must discover intrinsic geometric structure, dense regions, and natural groupings within unlabelled data.

---

## Chapter Objectives
- Understand unsupervised learning: discovering patterns without ground-truth target labels.
- Contrast three distinct clustering paradigms:
  1. **Centroid-Based**: Partitioning space around center prototypes ($K$-Means).
  2. **Density-Based**: Tracing continuous regions of high spatial density (DBSCAN).
  3. **Hierarchical**: Iteratively merging clusters bottom-up into a tree / dendrogram (Agglomerative).
- Understand why different algorithms make fundamentally different geometric assumptions (spherical vs. arbitrary shapes).
- Learn intrinsic cluster evaluation metrics (WCSS inertia, silhouette intuition) when no ground-truth labels exist.

---

## Prerequisites
Before tackling this chapter, ensure you have completed:
- **[Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing)** — Pairwise distance matrices, broadcasting, and boolean indexing.
- **[Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra)** — Euclidean distances, norms, and centroids (sample means).
- **[Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts)** — Supervised vs. unsupervised framing.

> [!NOTE]
> **Independent Curriculum Bridge**: Dimensionality reduction (Chapter 5) is **NOT** a prerequisite for clustering! You do not need PCA to perform clustering. Clustering operates directly on high- or low-dimensional coordinate spaces using distance metrics.

---

## 🧭 Pedagogical Progression & Intuition Bridges

To master clustering intuitively, absorb these stages:

### 1. Manual 2D Grouping (Pencil and Paper)
- Plot 4 points on a 2D plane: $A=(1, 1), B=(1, 2), C=(5, 5), D=(6, 5)$.
- Without running any code, your visual cortex immediately identifies two clusters: $\{A, B\}$ and $\{C, D\}$.
- Why? The distance within each group ($d(A, B) = 1$) is far smaller than the distance between groups ($d(B, C) = \sqrt{4^2 + 3^2} = 5$).
- **Centroids**: The center of $\{A, B\}$ is $(1, 1.5)$; the center of $\{C, D\}$ is $(5.5, 5)$.

### 2. Paradigm 1: Centroid-Based ($K$-Means)
- Alternates between two steps (Lloyd's Algorithm):
  1. **Assignment**: Assign each point to its nearest centroid.
  2. **Update**: Move each centroid to the mean of its assigned points.
- **Limitation**: Assumes clusters are convex and spherical. It cannot discover non-convex shapes like concentric circles or interlocking rings!

### 3. Paradigm 2: Density-Based (DBSCAN)
- Defines clusters as continuous, dense spatial regions separated by low-density noise.
- Evaluates points using an $\epsilon$-neighborhood and `min_samples` threshold:
  - **Core Point**: Has $\ge \text{min\_samples}$ within distance $\epsilon$.
  - **Border Point**: Within $\epsilon$ of a core point, but fewer than `min_samples` neighbors.
  - **Noise Point**: Isolated point belonging to no dense cluster (labeled $-1$).
- Can discover clusters of **arbitrary shape** (rings, crescents, spirals) and naturally filters outliers!

### 4. Paradigm 3: Hierarchical (Agglomerative)
- Starts with every point in its own individual cluster ($N$ clusters).
- Iteratively finds the two "closest" clusters and merges them until only $K$ clusters remain (or 1 root cluster forms a dendrogram).
- The definition of distance between two clusters determines the shape:
  - **Single Linkage**: Minimum distance between any two points (chains points, handles non-globular shapes).
  - **Complete Linkage**: Maximum distance (produces compact, spherical clusters).
  - **Average Linkage**: Average pairwise distance (robust compromise).

---

## Recommended Study Sequence

1. **[`kmeans.py`](./kmeans.py)** — Start here. Understand Lloyd's alternating minimization and K-Means++ smart initialization.
2. **[`dbscan.py`](./dbscan.py)** — Density-based clustering, breadth-first search expansion, and noise detection.
3. **[`agglomerative_clustering.py`](./agglomerative_clustering.py)** — Bottom-up hierarchical merging and linkage criteria.

---

## Chapter 4 Toy Datasets

- **Dataset K1 (Two Well-Separated 2D Clusters)**:
  - Cluster 1: $[[0, 0], [0, 1], [1, 0]]$
  - Cluster 2: $[[10, 10], [10, 11], [11, 10]]$
  - Hand check: Obvious gap. $K$-Means effortlessly assigns clusters 0 and 1 with zero misassignments.

- **Dataset K2 (Concentric Rings / Two Moons)**:
  - Inner circle of radius 1; outer ring of radius 5.
  - Hand check: $K$-Means cuts through the rings with a linear chord, failing completely. DBSCAN tracks the curved density and correctly separates the two rings.

- **Dataset K3 (Varying Density Clusters)**:
  - One dense compact cluster ($100$ points in radius $0.5$) and one sparse spread-out cluster ($100$ points in radius $5.0$).
  - Hand check: Illustrates where DBSCAN with a single fixed $\epsilon$ struggles, while Agglomerative clustering with Average Linkage succeeds.

---

## Completion Checklist
- [ ] Group a 4-point toy dataset and recompute centroids manually on paper.
- [ ] Implement `kmeans.py` with WCSS tracking and K-Means++ initialization.
- [ ] Implement `dbscan.py` with core/border/noise classification.
- [ ] Implement `agglomerative_clustering.py` with Single, Complete, and Average linkage.
- [ ] Test all three algorithms on Dataset K2 and compare their cluster boundaries.

---

## Cross-Chapter Conceptual Questions
1. **Connection to GMM (Chapter 7)**: How is $K$-Means a "hard assignment" special case of the Gaussian Mixture Model (where assignments are soft probabilities)?
2. **Deterministic vs Stochastic**: Why does $K$-Means produce different clusters depending on the random seed, whereas standard Agglomerative Clustering and DBSCAN are deterministic?
3. **Number of Clusters**: Why do $K$-Means and Agglomerative Clustering require specifying $K$ upfront, while DBSCAN automatically discovers the number of clusters?

---

## Personal Notes
*(Use this space to record your thoughts, notes, and reflections.)*
