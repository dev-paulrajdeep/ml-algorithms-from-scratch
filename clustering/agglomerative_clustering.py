"""
A. Mission
Agglomerative Clustering is a bottom-up hierarchical method. It starts with singletons and successively merges pairs until a single cluster remains, building a dendrogram. It allows cutting the tree at any level without prespecifying K.

B. Prerequisites
- [Stage 0C: Linear Algebra](../foundations/README.md#module-0c-linear-algebra): Norms and distance metrics.
- [Stage 0F: ML Core](../foundations/README.md#module-0f-machine-learning-core-concepts): Unsupervised framing.

C. Learning Questions
1. How do single, complete, and average linkage differ?
2. What is the "chaining effect" in single linkage?
3. Why might a hierarchical method be preferable for exploratory data analysis?

D. Mathematics to Derive
1. Write the formula for distance under Single Linkage: $\min_{a \in A, b \in B} d(a, b)$.
2. Derive formulas for Complete Linkage ($\max$) and Average Linkage ($\text{mean}$).
3. Observe how the distance matrix updates upon a merge.

E. Implementation Contract
- Class name: `AgglomerativeClustering`
- Parameters:
  - `n_clusters` (int)
  - `linkage` (str): 'single', 'complete', or 'average'
- Public Methods:
  - `fit(X)`
- Attributes:
  - `labels_` (shape `(N,)`)
  - `dendrogram_` (records merge history)

F. Guided Implementation Stages
**Checkpoint 1: Initial Distance Matrix**
- *What to learn*: Setting up the pairwise tracking.
- *What to do*: Compute an $N \times N$ distance matrix.
- *How to check yourself*: The diagonal should be zero, and the matrix symmetric.
- *When to proceed*: Distances are correctly captured.
- *Recovery hints*:
  - Hint 1: `scipy.spatial.distance.cdist` can be a good reference, but write your own loop or vectorized formula.
  - Hint 2: Be careful not to merge a cluster with itself; mask the diagonal.
  - Hint 3: Keep track of active cluster IDs.

**Checkpoint 2: The Merge Loop**
- *What to learn*: Hierarchical reduction.
- *What to do*: Find the minimum valid distance, merge the two clusters, record in `dendrogram_`, and update distances to the new cluster based on the `linkage` criterion. Stop when `n_clusters` remain or 1 single root is formed.
- *How to check yourself*: $N-1$ merges should occur if building the full tree.
- *When to proceed*: Correctly identifies merging clusters on Dataset 4A.
- *Recovery hints*:
  - Hint 1: Mask out inactive rows/cols with infinity instead of deleting them to preserve indexing.
  - Hint 2: Ensure your linkage update formulas correctly handle sizes for average linkage.
  - Hint 3: Track which original indices belong to which cluster ID for label extraction.

G. Edge Cases and Expected Tests
1. **Ties in Distance**: Can pick arbitrarily, but shouldn't crash.
2. **Two Points**: Perform exactly one merge.
3. **Identical Points**: Distance is 0, they should merge first.

H. Complexity Analysis
1. Time complexity of naive Agglomerative Clustering ($O(N^3)$ or $O(N^2 \log N)$)?
2. Space complexity required for the distance matrix ($O(N^2)$).

I. Progressive Difficulty Levels & Definition of Done
- Level 1 Guided: Implement the distance matrix and single linkage merge loop.
- Level 2 Standard: Add complete and average linkage, and `n_clusters` stopping.
- Level 3 Challenge: Implement full dendrogram history tracking and test against chaining effect on Dataset 4C.

J. Reflection
1. Why is this algorithm typically too slow for $N > 100,000$?
2. What are the advantages of not having to choose an initial random state?
"""
