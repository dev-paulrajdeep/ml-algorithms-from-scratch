r"""
A. **Mission** (Step 0: Understand the Problem)
$K$-Means is the canonical centroid-based unsupervised clustering algorithm. It partitions $N$ unlabelled data points into $K$ distinct clusters by minimizing the Within-Cluster Sum of Squares (WCSS), also known as **inertia**.
Real-world problem: Customer segmentation in e-commerce, color quantization in image compression, or automated geographic partitioning for delivery routing.
Because true labels do not exist, $K$-Means iteratively alternates between assigning each sample to its closest cluster centroid and recomputing centroids as the geometric mean of their assigned points.

B. **Prerequisites**
- [Stage 0B: NumPy & Numerical Computing](../foundations/README.md#module-0b-numpy-and-numerical-computing) — Array broadcasting, `np.argmin`, and boolean masks.
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra) — Euclidean norms ($\|\mathbf{x} - \boldsymbol{\mu}\|_2^2$) and sample centroids.
- [Stage 0F: Machine Learning Core](../foundations/README.md#module-0f-machine-learning-core-concepts) — Unsupervised learning framing and random seed reproducibility.

C. **Learning Questions** (Step 1: Build Intuition)
1. Why is Lloyd's algorithm guaranteed to converge? Why does the total WCSS objective function decrease (or stay constant) at every single assignment and update step?
2. Why is $K$-Means sensitive to initial centroid locations? How can poor random initialization trap the algorithm in a bad local minimum?
3. What geometric cluster shapes can $K$-Means discover (convex, spherical), and why does it fail on non-convex shapes like concentric circles (Dataset K2)?

D. **Mathematics to Derive**
1. **Step 2: Tiny Numerical Example by Hand**:
   Consider 4 points in 1D: $x = [1.0, 2.0, 8.0, 9.0]$ and $K = 2$.
   Initial random centroids: $\mu_1 = 0.0, \mu_2 = 10.0$.
   - **Assignment step**:
     Point 1 ($1.0$): distance to $\mu_1$ is $|1 - 0| = 1$; distance to $\mu_2$ is $|1 - 10| = 9 \implies \text{Cluster } 1$.
     Point 2 ($2.0$): distance to $\mu_1$ is $2$; to $\mu_2$ is $8 \implies \text{Cluster } 1$.
     Point 3 ($8.0$): distance to $\mu_1$ is $8$; to $\mu_2$ is $2 \implies \text{Cluster } 2$.
     Point 4 ($9.0$): distance to $\mu_1$ is $9$; to $\mu_2$ is $1 \implies \text{Cluster } 2$.
   - **Update step**:
     New centroid $\mu_1 = \frac{1 + 2}{2} = 1.5$.
     New centroid $\mu_2 = \frac{8 + 9}{2} = 8.5$.
   - Centroids moved to the exact true centers in a single iteration!
2. **Step 3: Mathematical Notation**:
   - $N$: number of samples, $D$: number of features, $K$: number of clusters.
   - $\mathbf{X} \in \mathbb{R}^{N \times D}$: unlabelled dataset.
   - $\boldsymbol{\mu}_k \in \mathbb{R}^D$: center (centroid) of cluster $k$.
   - $r_{ik} \in \{0, 1\}$: binary indicator where $r_{ik} = 1$ if sample $i$ belongs to cluster $k$.
3. **Step 4: Derive the Objective Function and Minimizers**:
   - Write the Within-Cluster Sum of Squares (Inertia):
     $$J = \sum_{i=1}^N \sum_{k=1}^K r_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$$
   - **Step 1 (Assignment)**: Holding $\boldsymbol{\mu}_k$ fixed, minimize $J$ with respect to $r_{ik}$.
     Since terms for each $i$ are decoupled, $J$ is minimized by assigning each $\mathbf{x}_i$ to the centroid with minimal distance:
     $$r_{ik} = \begin{cases} 1 & \text{if } k = \arg\min_j \|\mathbf{x}_i - \boldsymbol{\mu}_j\|_2^2 \\ 0 & \text{otherwise} \end{cases}$$
   - **Step 2 (Update)**: Holding $r_{ik}$ fixed, minimize $J$ with respect to $\boldsymbol{\mu}_k$.
     Take the derivative $\frac{\partial J}{\partial \boldsymbol{\mu}_k} = -2 \sum_{i=1}^N r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k) = \mathbf{0}$:
     $$\boldsymbol{\mu}_k = \frac{\sum_{i=1}^N r_{ik} \mathbf{x}_i}{\sum_{i=1}^N r_{ik}} = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$$
     The geometric mean mathematically minimizes the sum of squared distances!
   - **$K$-Means++ Initialization Rule**:
     1. Choose first centroid $\boldsymbol{\mu}_1$ uniformly at random from $\mathbf{X}$.
     2. For each remaining point $\mathbf{x}_i$, compute $D(\mathbf{x}_i) = \min_{j} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|_2$.
     3. Choose next centroid with probability proportional to squared distance: $P(\mathbf{x}_i) = \frac{D(\mathbf{x}_i)^2}{\sum_m D(\mathbf{x}_m)^2}$.

E. **Implementation Contract**
- Class: `KMeans`
- Constructor:
  - `__init__(self, k: int = 3, init: str = 'kmeans++', max_iter: int = 100, tol: float = 1e-4)`
- Attributes:
  - `centroids_`: array of shape `(k, D)`.
  - `labels_`: array of shape `(N,)` with cluster assignments in $\{0, \dots, k-1\}$.
  - `inertia_`: float, total WCSS on training data.
- Methods:
  - `fit(self, X: np.ndarray) -> self`:
    Learns centroids and assigns training labels until centroid shift $< tol$ or `max_iter` reached.
  - `predict(self, X: np.ndarray) -> np.ndarray`:
    Assigns each query vector in $X$ to its closest centroid in `centroids_`. Returns shape `(N,)`.

F. **Guided Implementation Stages**
**Step 5: Design the Algorithm (Pseudocode)**
```text
fit(X):
  centroids = initialize_centroids(X, k, init)
  For iteration in range(max_iter):
    # Assignment: compute distances to all centroids
    distances = compute_pairwise_distances(X, centroids) # (N, k)
    labels = argmin(distances, axis=1)
    
    # Update: recompute centroids
    new_centroids = array([mean(X[labels == j], axis=0) for j in range(k)])
    
    # Check convergence
    shift = norm(new_centroids - centroids)
    centroids = new_centroids
    if shift < tol:
      break
  store centroids_, labels_, inertia_
```

**Checkpoint 1: Initialization (`random` vs `kmeans++`)**
- **What to learn**: Seeding initial cluster prototypes.
- **What to do**: Implement random sampling without replacement (`np.random.choice(N, size=k, replace=False)`).
- **How to check yourself**: Initial centroids have shape `(k, D)` and represent actual data points.
- **When to proceed**: Initialization returns distinct coordinates.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Initializing all centroids to $(0, 0)$ will trap the algorithm in a single center.
  - *Hint 2 (Operation)*: Index `X[indices]` with randomly selected row indices.
  - *Hint 3 (Debugging)*: Set random seeds for reproducible testing.

**Checkpoint 2: Distance Matrix & Assignment**
- **What to learn**: Vectorized computation of distances from $N$ points to $K$ centroids.
- **What to do**: Compute distances using broadcasting: `np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)`.
- **How to check yourself**: Output matrix has shape `(N, k)`. Taking `np.argmin(..., axis=1)` yields assignments in $\{0, \dots, k-1\}$.
- **When to proceed**: On Dataset K1, points are assigned to the correct cluster center.
- **Recovery hints**:
  - *Hint 1 (Concept)*: Shape `(N, 1, D)` minus shape `(1, K, D)` broadcasts to `(N, K, D)`.
  - *Hint 2 (Operation)*: Alternatively, use a loop over $k$: `dist[:, j] = np.linalg.norm(X - centroids[j], axis=1)`.
  - *Hint 3 (Debugging)*: Ensure `axis=1` when taking `argmin`.

**Checkpoint 3: Centroid Recomputation & Convergence**
- **What to learn**: Updating means and checking stopping conditions.
- **What to do**: For each cluster $j$, compute `new_centroids[j] = X[labels == j].mean(axis=0)`. Handle empty clusters by reassigning to the point with maximum inertia.
- **How to check yourself**: WCSS inertia decreases or stays constant at each iteration.
- **When to proceed**: Algorithm halts when centroid shift $< tol$ and recovers known clusters on Dataset K1.
- **Recovery hints**:
  - *Hint 1 (Concept)*: If a cluster becomes empty (`len(X[labels == j]) == 0`), assigning `X.mean(axis=0)` or a random point prevents `NaN`.
  - *Hint 2 (Operation)*: Compute shift as `np.linalg.norm(new_centroids - old_centroids)`.
  - *Hint 3 (Debugging)*: Compute total inertia as `sum((X - centroids[labels])**2)`.

G. **Edge Cases and Expected Tests** (Step 8: Test and Debug)
1. `test_two_separated_clusters_dataset_k1`: $K$-Means must perfectly cluster Dataset K1 with zero misassignments.
2. `test_single_cluster_k_equals_1`: When $K=1$, the learned centroid must equal `np.mean(X, axis=0)`.
3. `test_empty_cluster_handling`: Construct an artificial initialization where one centroid receives 0 points; verify the model does not crash or produce `NaN`.
4. `test_inertia_monotonicity`: Across all iterations, the recorded inertia must be strictly non-increasing.

H. **Complexity Analysis** (Step 9: Analyze the Algorithm)
- **Time Complexity**:
  - Distance computation per iteration: $O(N \cdot K \cdot D)$ time.
  - Centroid update per iteration: $O(N \cdot D)$ time.
  - Total fitting: $O(\text{iterations} \cdot N \cdot K \cdot D)$.
  - Inference: $O(N_{\text{test}} \cdot K \cdot D)$ time.
- **Space Complexity**: $O(K \cdot D)$ auxiliary memory for centroids.
- **Trade-offs**: Fast, scalable, and simple, but restricted to spherical clusters and sensitive to initialization and outliers.

I. **Progressive Difficulty Levels & Definition of Done**
- **Level 1 (Guided)**: Implement basic $K$-Means with random initialization on 2D Dataset K1.
- **Level 2 (Standard)**: Complete `KMeans` class with $K$-Means++ initialization, empty cluster handling, and tolerance convergence.
- **Level 3 (Challenge)**: Implement multiple random restarts (`n_init = 10`), returning the fit with minimal inertia, and write a helper for the **Elbow Method** to plot inertia vs. $K$.
- **Definition of Done**: Converges reliably, achieves monotonic WCSS decrease, implements $K$-Means++, and handles degenerate empty clusters gracefully.

J. **Reflection & Self-Check** (Step 10: Independent Completion)
1. Why does feature standardization (zero mean, unit variance) drastically change the cluster assignments of $K$-Means?
2. Why is $K$-Means unable to separate the concentric rings in Dataset K2, even with $K=2$?
"""
