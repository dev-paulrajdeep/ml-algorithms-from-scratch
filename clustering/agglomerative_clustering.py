r"""
A. **Mission** (Step 0: Understand the Problem)
Hierarchical Agglomerative Clustering is a bottom-up unsupervised clustering algorithm. It begins by treating each individual data point as a singleton cluster ($N$ initial clusters) and iteratively merges the two closest clusters according to a chosen **linkage criterion** until a stopping condition is met (e.g., exactly $K$ clusters remain).
Real-world problem: Constructing biological phylogenetic trees from genetic distances, topic hierarchies in document clustering, or organizational structures.
Unlike $K$-Means, Agglomerative Clustering produces a complete nested hierarchy (a **dendrogram**), does not require random initialization, and allows you to explore clusterings at any level of granularity without re-running the algorithm.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — 2D pairwise distance matrices and matrix masking.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Euclidean distances and metric spaces.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Unsupervised learning framing.
- [K-Means](./kmeans.py) — Understanding flat partitioning as contrast.

C. **Learning Questions** (Step 1: Build Intuition)
1. How does the choice of **linkage criterion** fundamentally change the shapes of clusters discovered?
   - What is the "chaining effect" in **Single Linkage**, and why does it produce long, thin string-like clusters?
   - Why does **Complete Linkage** favor compact, spherical clusters of roughly equal diameter?
2. What is a **dendrogram**, and how does looking at the vertical branch lengths help you decide the natural number of clusters in a dataset?
3. Why is standard Agglomerative Clustering completely deterministic (yielding the exact same clusters every run), unlike $K$-Means?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider 4 points in 1D: $x = [1.0, 2.0, 5.0, 9.0]$.
   Initial clusters: $C_0=\{1\}, C_1=\{2\}, C_2=\{5\}, C_3=\{9\}$.
   Pairwise distances:
   $d(C_0, C_1) = 1.0$, $d(C_1, C_2) = 3.0$, $d(C_2, C_3) = 4.0$.
   - **Merge 1**: Minimum distance is $d(C_0, C_1) = 1.0$. Merge $C_0$ and $C_1 \implies C_4 = \{1, 2\}$.
   - Remaining clusters: $C_4=\{1, 2\}, C_2=\{5\}, C_3=\{9\}$.
   - Under **Single Linkage**: $d(C_4, C_2) = \min(d(1, 5), d(2, 5)) = \min(4, 3) = 3.0$.
   - Under **Complete Linkage**: $d(C_4, C_2) = \max(d(1, 5), d(2, 5)) = \max(4, 3) = 4.0$.
   - **Merge 2** (Single Linkage): Minimum distance is $d(C_4, C_2) = 3.0$. Merge $C_4$ and $C_2 \implies C_5 = \{1, 2, 5\}$.
   - When $K=2$, remaining clusters are $\{1, 2, 5\}$ and $\{9\}$.
2. **Step 3: Mathematical Notation**:
   - $N$: number of samples, $K$: desired target clusters (`n_clusters`).
   - Clusters $A$ and $B$, containing points $\mathbf{a} \in A$ and $\mathbf{b} \in B$.
   - $d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2$: Euclidean point distance.
   - $D(A, B)$: distance between clusters $A$ and $B$.
3. **Step 4: Formalize the Linkage Criteria**:
   - **Single Linkage** (Minimum Distance):
     $$D_{\text{single}}(A, B) = \min_{\mathbf{a} \in A, \mathbf{b} \in B} d(\mathbf{a}, \mathbf{b})$$
   - **Complete Linkage** (Maximum Distance):
     $$D_{\text{complete}}(A, B) = \max_{\mathbf{a} \in A, \mathbf{b} \in B} d(\mathbf{a}, \mathbf{b})$$
   - **Average Linkage** (Mean Pairwise Distance / UPGMA):
     $$D_{\text{average}}(A, B) = \frac{1}{|A| |B|} \sum_{\mathbf{a} \in A} \sum_{\mathbf{b} \in B} d(\mathbf{a}, \mathbf{b})$$

E. **Implementation Contract**
- Class: `AgglomerativeClustering`
- Constructor:
  - `__init__(self, n_clusters: int = 2, linkage: str = 'single')`:
    Stores hyperparameters. Validates `linkage in ('single', 'complete', 'average')`.
- Attributes:
  - `labels_`: array of shape `(N,)` with integer cluster assignments in $\{0, \dots, n_clusters - 1\}$.
  - `children_`: optional list or array recording the $N - n_clusters$ merge operations.
- Methods:
  - `fit(self, X: np.ndarray) -> self`:
    Builds the pairwise distance matrix, iteratively merges the closest clusters until `n_clusters` remain, and maps final cluster IDs to `labels_`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X):
  N = len(X)
  clusters = {i: [i] for i in range(N)} # active cluster id -> list of original point indices
  dist_matrix = compute_pairwise_distances(X) # (N, N)
  fill diagonal of dist_matrix with infinity
  
  While len(clusters) > n_clusters:
    # Find closest pair of active clusters
    (c1, c2) = argmin(dist_matrix)
    
    # Merge c2 into c1
    clusters[c1].extend(clusters[c2])
    del clusters[c2]
    
    # Update dist_matrix row/col for c1 using chosen linkage
    For each remaining cluster c in clusters:
      dist_matrix[c1, c] = compute_linkage(X[clusters[c1]], X[clusters[c]], linkage)
      dist_matrix[c, c1] = dist_matrix[c1, c]
      
    # Invalidate c2 row/col with infinity
    dist_matrix[c2, :] = inf
    dist_matrix[:, c2] = inf

  # Assign clean labels 0 to n_clusters-1
  labels = zeros(N, dtype=int)
  For new_id, (old_id, point_indices) in enumerate(clusters.items()):
    labels[point_indices] = new_id
  store labels_
```

**Checkpoint 1: Initial Distance Matrix**
- **What to learn**: Initializing $N \times N$ pairwise distance representation.
- **What to do**: Compute Euclidean distances between all pairs of samples. Set the diagonal to $\infty$ so clusters do not merge with themselves.
- **How to check yourself**: Matrix is symmetric, shape `(N, N)`, and diagonal values are `np.inf`.
- **When to proceed**: Distance matrix passes sanity checks.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Setting diagonal to `np.inf` avoids finding $d(i, i) = 0$ as the minimum.
  - *Hint 2 (Operation)*: `dists = np.linalg.norm(X[:, None] - X[None, :], axis=2); np.fill_diagonal(dists, np.inf)`.
  - *Hint 3 (Debugging)*: Ensure matrix is symmetric: `assert np.allclose(dists, dists.T)`.

**Checkpoint 2: Closest Pair Selection and Merge**
- **What to learn**: Finding minimum distance coordinates in 2D array.
- **What to do**: Use `np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)` to find indices $(c_1, c_2)$.
- **How to check yourself**: Coordinates $c_1 \neq c_2$ and have minimal finite distance.
- **When to proceed**: Successfully identifies the closest pair on a 4-point toy array.
- **Recovery hints**:
  - *Hint 1 (Concept)*: `np.argmin` flattens the matrix; `np.unravel_index` converts back to 2D row and column.
  - *Hint 2 (Operation)*: If $c_1 > c_2$, swap them so you merge into the smaller index.
  - *Hint 3 (Debugging)*: Check that the minimum distance is strictly finite.

**Checkpoint 3: Linkage Distance Updating**
- **What to learn**: Updating inter-cluster distances after a merge.
- **What to do**: When $c_1$ and $c_2$ merge into $c_1$:
  - Single: `new_dist = min(dist_matrix[c1, k], dist_matrix[c2, k])`.
  - Complete: `new_dist = max(dist_matrix[c1, k], dist_matrix[c2, k])`.
  - Average: compute mean pairwise distance between all points in $c_1$ and $k$.
- **How to check yourself**: Distance matrix updates properly and reduces cluster count by 1 each step.
- **When to proceed**: Merging terminates when exactly `n_clusters` remain.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Single and Complete linkage can update in $O(1)$ by taking min or max of existing rows!
  - *Hint 2 (Operation)*: Mask out deleted row/col $c_2$ by setting all entries to `np.inf`.
  - *Hint 3 (Debugging)*: Map final active clusters to clean contiguous integers $\{0, \dots, n\_clusters - 1\}$.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_n_clusters_equals_n_samples`: When `n_clusters = N`, no merges occur; each point gets its own unique label $0$ to $N-1$.
2. `test_n_clusters_equals_1`: Merging continues until all points belong to a single cluster 0.
3. `test_single_vs_complete_linkage_behavior`: Construct a dataset with two elongated chaining clusters. Show that Single Linkage preserves the chain, whereas Complete Linkage splits it into compact spheres.
4. `test_two_well_separated_clusters_dataset_k1`: Algorithm cleanly recovers the two clusters on Dataset K1.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Naive implementation: $N-K$ merge steps, each scanning an $N \times N$ matrix $\implies O(N^3)$ time.
  - Priority Queue / Heap optimization: $O(N^2 \log N)$ time.
- **Space Complexity**: $O(N^2)$ memory for the pairwise distance matrix.
- **Trade-offs**: Deterministic and produces an informative hierarchical tree, but $O(N^2)$ memory and $O(N^3)$ time make it impractical for massive datasets ($N > 50,000$).

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement distance matrix and Single Linkage merge loop on a 4-point toy dataset.
- **Level 2 (Standard)**: Complete `AgglomerativeClustering` supporting Single, Complete, and Average linkage criteria, stopping at `n_clusters`.
- **Level 3 (Challenge)**: Track the full merge history (`children_` array recording merged cluster IDs and distances) to reconstruct the dendrogram.
- **Definition of Done**: Cleanly supports all three linkages, assigns contiguous cluster labels, and passes all edge tests.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does Single Linkage suffer from the "chaining effect" (where clusters can stretch across noisy outlier bridges)?
2. What are the advantages of not having to specify an initial random seed in Agglomerative Clustering?
"""
