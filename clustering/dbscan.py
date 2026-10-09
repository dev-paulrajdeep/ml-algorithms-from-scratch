"""
A. Mission
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups closely packed points and marks solitary points in low-density regions as noise. It finds arbitrarily shaped clusters without specifying K.

B. Prerequisites
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra): Norms and distance metrics.
- [Stage 0F: ML Core](../foundations/README.md#module-0f-machine-learning-core-concepts): Unsupervised framing.

C. Learning Questions
1. How are core, border, and noise points defined?
2. What is density-reachability?
3. Why does DBSCAN succeed on Dataset 4B (concentric rings) while K-Means fails?

D. Mathematics to Derive
1. Formulate the definition of the $\epsilon$-neighborhood: $N_\epsilon(p) = \{q \in D | \text{dist}(p, q) \le \epsilon\}$.
2. Prove that if $p$ is a core point and $q$ is density-reachable from $p$, they are in the same cluster.

E. Implementation Contract
- Class name: `DBSCAN`
- Parameters:
  - `eps` (float)
  - `min_samples` (int)
- Public Methods:
  - `fit(X)`
- Attributes:
  - `labels_` (shape `(N,)`, with -1 for noise)

F. Guided Implementation Stages
**Checkpoint 1: Neighborhood Query**
- *What to learn*: Finding points within a specific distance threshold.
- *What to do*: Implement a function to return indices of all points within `eps` of point `p`.
- *How to check yourself*: Verify on a small mock dataset that exact counts match expectations.
- *When to proceed*: Distances are correctly calculated and thresholded.
- *Recovery hints*:
  - Hint 1: Vectorized distance calculation helps speed things up.
  - Hint 2: A point is always in its own neighborhood.
  - Hint 3: Use `np.linalg.norm`.

**Checkpoint 2: Core vs Noise Classification and Expansion**
- *What to learn*: Expanding clusters via density reachability.
- *What to do*: Iterate points, start a cluster if core, use BFS/DFS to expand, mark as noise (-1) otherwise.
- *How to check yourself*: Run on Dataset 4B (moons); it should assign one cluster to each moon and -1 to far-away outliers.
- *When to proceed*: The algorithm finishes without infinite loops and assigns valid labels.
- *Recovery hints*:
  - Hint 1: Keep an array for visited status.
  - Hint 2: If a noise point is later reached during expansion, it becomes a border point (change its label, but don't expand its neighborhood).
  - Hint 3: Ensure your queue/stack doesn't re-process points endlessly.

G. Edge Cases and Expected Tests
1. **Empty Clusters / All Noise**: Very small `eps` should result in all points being -1.
2. **Single Cluster**: Huge `eps` groups everything into cluster 0.
3. **Identical Points**: Should mutually include each other in their neighborhoods.

H. Complexity Analysis
1. What is the naive time complexity ($O(N^2)$)?
2. How does a spatial index (KD-Tree) improve this?
3. Space complexity requirements?

I. Progressive Difficulty Levels & Definition of Done
- Level 1 Guided: Implement neighborhood queries and simple core/noise loops.
- Level 2 Standard: Full cluster expansion (BFS) ensuring border points are assigned correctly.
- Level 3 Challenge: Optimize distance computations and test on Dataset 4C varying densities.

J. Reflection
1. Why does a single `eps` struggle with Dataset 4C (varying density)?
2. In what situations might border point assignments be non-deterministic?
"""
