r"""
A. **Mission** (Step 0: Understand the Problem)
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised clustering algorithm that discovers clusters of arbitrary geometric shape by identifying contiguous regions of high point density separated by regions of low density.
Real-world problem: Spatial anomaly detection (detecting GPS fraud or fraudulent credit transactions), astronomy (identifying star clusters against cosmic background noise), or clustering complex non-linear spatial trajectories.
Unlike $K$-Means, DBSCAN **does not require specifying the number of clusters $K$ in advance** and is immune to being tricked by non-convex shapes like concentric circles or spirals. Crucially, it designates isolated points as **noise** (label $-1$) rather than forcing them into artificial clusters.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — 2D pairwise distance matrices, boolean indexing, queues/sets.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Euclidean distance and metric spaces.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Unsupervised learning framing.
- [K-Means](./kmeans.py) — Understanding centroid-based limitations on non-convex shapes.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why does DBSCAN successfully separate concentric circles (Dataset K2) while $K$-Means completely fails?
2. How do the two hyperparameters `eps` ($\epsilon$) and `min_samples` define the density threshold? What happens if `eps` is too small (everything is noise) vs. too large (everything merges into one giant cluster)?
3. What is the fundamental difference between a **core point**, a **border point**, and a **noise point**?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider 5 points in 1D: $x = [1.0, 1.2, 1.4, 5.0, 10.0]$ with $\epsilon = 0.5$ and $\text{min\_samples} = 3$.
   - Point 0 ($1.0$): Neighbors within $0.5$ are $\{1.0, 1.2, 1.4\}$ (count = 3). Since $3 \ge 3$, this is a **Core Point**!
   - Point 1 ($1.2$): Neighbors are $\{1.0, 1.2, 1.4\}$ (count = 3). **Core Point**!
   - Point 2 ($1.4$): Neighbors are $\{1.0, 1.2, 1.4\}$ (count = 3). **Core Point**!
   - Point 3 ($5.0$): Neighbors are $\{5.0\}$ (count = 1). Since $1 < 3$, and not near any core point, this is **Noise** (label $-1$).
   - Point 4 ($10.0$): Neighbors are $\{10.0\}$ (count = 1). **Noise** (label $-1$).
   - Result: One cluster $\{0, 1, 2\}$ and two noise points $\{3, 4\}$.
2. **Step 3: Mathematical Notation**:
   - $N$: number of samples, $D$: number of features.
   - $\epsilon \in \mathbb{R}^+$: radius of the neighborhood ball.
   - $\text{min\_samples} \in \mathbb{Z}^+$: minimum point count to constitute high density.
   - $N_\epsilon(\mathbf{p}) = \{\mathbf{q} \in \mathbf{X} : \|\mathbf{p} - \mathbf{q}\|_2 \le \epsilon\}$: $\epsilon$-neighborhood of point $\mathbf{p}$.
3. **Step 4: Formalize the Density Connectivity Definitions**:
   - **Core Point**: A point $\mathbf{p}$ is a core point if $|N_\epsilon(\mathbf{p})| \ge \text{min\_samples}$.
   - **Directly Density-Reachable**: Point $\mathbf{q}$ is directly density-reachable from $\mathbf{p}$ if $\mathbf{q} \in N_\epsilon(\mathbf{p})$ and $\mathbf{p}$ is a core point.
   - **Density-Reachable**: A point $\mathbf{q}$ is density-reachable from $\mathbf{p}$ if there exists a chain of points $\mathbf{p}_1, \dots, \mathbf{p}_k$ such that $\mathbf{p}_1 = \mathbf{p}, \mathbf{p}_k = \mathbf{q}$, and each $\mathbf{p}_{i+1}$ is directly density-reachable from $\mathbf{p}_i$.
   - **Density-Connected**: Points $\mathbf{p}$ and $\mathbf{q}$ are density-connected if there exists a point $\mathbf{o}$ such that both $\mathbf{p}$ and $\mathbf{q}$ are density-reachable from $\mathbf{o}$.
   - *Cluster Definition*: A maximal set of density-connected points.

E. **Implementation Contract**
- Class: `DBSCAN`
- Constructor:
  - `__init__(self, eps: float = 0.5, min_samples: int = 5)`:
    Stores hyperparameters.
- Attributes:
  - `labels_`: array of shape `(N,)` with integer cluster IDs ($0, 1, \dots$) or `-1` for noise.
  - `core_sample_indices_`: array of indices of points identified as core points.
- Methods:
  - `fit(self, X: np.ndarray) -> self`:
    Identifies core points, expands clusters using Breadth-First Search (BFS) or Depth-First Search (DFS), and sets `labels_`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X):
  N = len(X)
  labels = full(N, -1) # initialize all as noise (-1)
  visited = zeros(N, dtype=bool)
  cluster_id = 0
  
  For i in range(N):
    if visited[i]:
      continue
    visited[i] = True
    neighbors = region_query(X, i, eps)
    
    if len(neighbors) < min_samples:
      labels[i] = -1 # marked as noise (may become border later)
    else:
      # Expand cluster using BFS queue
      labels[i] = cluster_id
      queue = list(neighbors)
      while queue is not empty:
        j = queue.pop(0)
        if not visited[j]:
          visited[j] = True
          j_neighbors = region_query(X, j, eps)
          if len(j_neighbors) >= min_samples:
            queue.extend(j_neighbors) # core point expands the frontier
        if labels[j] == -1:
          labels[j] = cluster_id # border or core point joins cluster
      cluster_id += 1
  store labels_
```

**Checkpoint 1: Neighborhood Query Helper**
- **What to learn**: Finding indices of all points within distance $\epsilon$.
- **What to do**: Implement `_region_query(X, point_idx, eps)`. Compute Euclidean distances from `X[point_idx]` to all points and return indices where `distance <= eps`.
- **How to check yourself**: The query must always include `point_idx` itself (distance $= 0.0 \le \epsilon$).
- **When to proceed**: Returns correct neighbor sets on tiny 5-point examples.
- **Recovery hints**:
  - *Hint 1 (Concept)*: A point is always a member of its own $\epsilon$-neighborhood.
  - *Hint 2 (Operation)*: `dists = np.linalg.norm(X - X[point_idx], axis=1); return np.where(dists <= eps)[0]`.
  - *Hint 3 (Debugging)*: Use `<=` rather than `<` so boundary points at exact distance $\epsilon$ are included.

**Checkpoint 2: Core Point Classification**
- **What to learn**: Determining which points have sufficient density.
- **What to do**: A point is a core point if `len(neighbors) >= min_samples`.
- **How to check yourself**: On Dataset K1, points in dense cluster centers are core points; isolated points outside are not.
- **When to proceed**: Correctly distinguishes core points from border/noise points.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Core points form the "backbone" of a cluster.
  - *Hint 2 (Operation)*: Store boolean array `is_core = np.zeros(N, dtype=bool)`.
  - *Hint 3 (Debugging)*: Remember `min_samples` includes the query point itself!

**Checkpoint 3: Cluster Expansion with Queue (BFS)**
- **What to learn**: Tracing density-connected components without infinite recursion.
- **What to do**: When a core point is found, initialize a new `cluster_id`. Use a queue to visit neighbors. If a neighbor is also a core point, add its neighbors to the queue. Assign `labels[neighbor] = cluster_id`.
- **How to check yourself**: Points in the same cluster receive identical positive integers. Outliers remain `-1`.
- **When to proceed**: On concentric rings (Dataset K2), outer ring receives label 0, inner circle receives label 1, and neither is split.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Border points join the cluster, but their neighbors are NOT added to the queue (they cannot expand the cluster).
  - *Hint 2 (Operation)*: Avoid duplicate additions to the queue by tracking `visited` status.
  - *Hint 3 (Debugging)*: If a point was previously labeled as noise (`-1`), change its label to `cluster_id` when reached by a core point (it is a border point!).

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_concentric_circles_dataset_k2`: DBSCAN must separate the two non-convex concentric rings where $K$-Means fails.
2. `test_isolated_noise_points`: Add several far-flung outlier points to Dataset K1. Verify that they receive label `-1`.
3. `test_epsilon_too_small`: When $\epsilon$ is smaller than the minimum pairwise distance, all points must be labeled as noise (`labels_ == -1`).
4. `test_epsilon_too_large`: When $\epsilon$ is larger than the maximum pairwise distance, all points must merge into a single cluster (`labels_ == 0`).

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Pairwise distance calculation: $O(N^2 \cdot D)$ time without spatial indexing.
  - Cluster traversal: $O(N)$ queue operations.
  - Total time: $O(N^2 \cdot D)$ naive; reducible to $O(N \log N)$ with KD-Tree or Ball-Tree in low dimensions ($D \le 10$).
- **Space Complexity**: $O(N)$ for visited tracking and labels (or $O(N^2)$ if precomputing full distance matrix).
- **Trade-offs**: Finds arbitrary shapes and filters noise, but struggles with datasets containing clusters of widely varying densities (Dataset K3).

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement distance query and identify core points on a 1D toy array.
- **Level 2 (Standard)**: Complete `DBSCAN` class with BFS queue expansion, border point handling, and noise label assignments.
- **Level 3 (Challenge)**: Optimize neighbor search by precomputing the pairwise distance matrix using vectorized matrix multiplication, and explore sensitivity to $\epsilon$ on Dataset K3.
- **Definition of Done**: Successfully clusters Dataset K1 and non-convex Dataset K2, labels outliers as $-1$, and passes all edge tests.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does DBSCAN struggle when clusters have dramatically different densities (e.g., one dense cluster and one sparse cluster)?
2. In what situations can border point cluster assignment be non-deterministic (depending on the visitation order of points)?
"""
