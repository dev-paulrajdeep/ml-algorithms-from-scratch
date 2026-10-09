# Foundations: Machine Learning Prerequisites

Welcome to the Stage 0 of the ML-from-scratch curriculum! This foundational stage provides the entire mathematical and programming on-ramp for the course. 

If you are a complete beginner, start from Module 0A. If you already have experience, use the Self-Assessment Checklists at the end of each module to determine if you can safely skip ahead.

## Roadmap Diagram

```text
[Module 0A: Programming] --> [Module 0B: NumPy]
                                 |
                                 v
[Module 0C: Linear Algebra] --> [Module 0F: ML Core Concepts] <-- [Module 0E: Probability & Statistics]
                                 ^
                                 |
                    [Module 0D: Calculus & Opt]

All modules --> [The Experimental Workflow] --> [Stage 0 Exit Self-Assessment] --> [Proceed to Chapter 1]
```

---

## <a name="module-0a-programming-readiness"></a>Module 0A: Programming Readiness

Before writing machine learning algorithms, you must be comfortable writing basic Python code.

### 1. Variables, Types, Expressions, Conditionals, Loops
**Explanation**: Python stores data in variables. Basic types include integers (`int`), floating-point numbers (`float`), strings (`str`), and booleans (`bool`). Expressions compute values. Conditionals (`if`/`elif`/`else`) control flow based on boolean logic. Loops (`for`/`while`) repeat operations.
**Worked Example**:
```python
x = 5          # int
y = 2.5        # float
is_greater = x > y  # bool (True)
if is_greater:
    for i in range(3):
        print(f"Loop {i}, x + y = {x + y}")
```
**Practice Task**: Write a loop that prints the square of even numbers from 1 to 10.
**Common Mistakes**: Off-by-one errors in `range(1, 10)` (stops at 9). Mixing tabs and spaces for indentation.
**Checkpoint**: Can you predict the output of the example above without running it?

### 2. Functions, Parameters, Return, Scope
**Explanation**: Functions are reusable blocks of code. They take inputs (parameters), perform actions, and give back outputs (`return`). Scope dictates where a variable is accessible (local inside a function, global outside).
**Worked Example**:
```python
def calculate_area(width, height):
    area = width * height  # 'area' is locally scoped
    return area

total = calculate_area(5, 4)
print(total) # 20
```
**Practice Task**: Write a function `is_even(n)` that returns `True` if `n` is even, and `False` otherwise.
**Common Mistakes**: Forgetting the `return` statement (function returns `None`). Trying to access a local variable from outside the function.
**Checkpoint**: Do you understand why `print(area)` outside the function would cause a NameError?

### 3. Lists, Dicts, Iteration, Error Handling
**Explanation**: Lists are ordered, mutable collections of items. Dictionaries are key-value pairings for fast lookups. Iteration allows you to traverse these structures. Error handling (`try`/`except`) prevents your program from crashing on expected errors.
**Worked Example**:
```python
data = [10, 20, 30]
hyperparams = {"learning_rate": 0.01, "epochs": 100}

try:
    print(hyperparams["momentum"])
except KeyError:
    print("Momentum not found, using default 0.9")
```
**Practice Task**: Iterate over a dictionary of fruits and prices, printing only fruits that cost more than $2.
**Common Mistakes**: Modifying a list while iterating over it. Catching broad `Exception` instead of specific errors like `KeyError` or `ValueError`.
**Checkpoint**: Can you iterate through both keys and values of a dictionary using `.items()`?

### 4. Modules, Imports, Tracebacks
**Explanation**: Python code is organized into modules. You use `import` to use code from other files. When errors occur, Python prints a traceback showing the sequence of function calls that led to the crash.
**Worked Example**:
```python
import math
print(math.sqrt(16)) # 4.0
```
**Practice Task**: Import the `random` module and generate a random integer between 1 and 10 using `random.randint()`.
**Common Mistakes**: Circular imports (A imports B, B imports A). Not reading the bottom of a traceback first to find the actual error message.
**Checkpoint**: Do you know how to read a traceback from bottom to top?

### 5. Basic Testing and Debugging
**Explanation**: Testing ensures your code does what you expect. Debugging is the process of finding and fixing bugs. `assert` statements are a simple way to test conditions.
**Worked Example**:
```python
def add(a, b): return a + b

# A simple test
assert add(2, 3) == 5, "Addition failed!"
```
**Practice Task**: Write an `assert` statement that checks if the length of the list `[1, 2, 3]` is 3.
**Common Mistakes**: Using print statements for testing instead of systematic assertions. 
**Checkpoint**: Do you know how to insert `print()` statements to check variable states before a crash?

### 6. Python Classes (Fit/Predict API Pattern)
**Explanation**: Classes group data (state) and functions (methods) together. In ML, we often use an API where a class has an `__init__` method for hyperparameters, a `fit` method for training, and a `predict` method for inference.
**Worked Example**:
```python
class DummyModel:
    def __init__(self, constant=0):
        self.constant = constant
        
    def fit(self, X, y):
        # Pretend to learn a parameter
        self.learned_param_ = self.constant + 1
        return self
        
    def predict(self, X):
        return [self.learned_param_ for _ in X]
```
**Practice Task**: Create a `MultiplierModel` class that takes a `multiplier` in `__init__`, saves it, and has a `predict(X)` method that returns each element in `X` multiplied by `multiplier`.
**Common Mistakes**: Forgetting `self` as the first argument in method definitions. Forgetting to initialize instance variables.
**Checkpoint**: Can you explain the difference between a class and an instance (object) of that class?

#### Self-Assessment Checklist: Module 0A
- [ ] **Skills**: Can write functions, classes, loops, conditionals, and handle exceptions.
- [ ] **Conceptual**: Explain local vs global scope.
- [ ] **Hand Calculations**: Trace the execution of a nested loop by hand.
- [ ] **Practice Task**: Implement a class with `fit` and `predict` that remembers a single value passed to `fit`.
- **Exit Criterion**: You can write a Python script from scratch that reads data into a dictionary, processes it with a function, and catches missing keys safely.
- **Recovery Recommendation**: Revisit a beginner Python tutorial (e.g., Python documentation tutorial) if concepts like loops or dictionaries feel confusing.

---

## <a name="module-0b-numpy-and-numerical-computing"></a>Module 0B: NumPy and Numerical Computing

NumPy is the foundational library for numerical computing in Python. It provides the `ndarray` object and optimized operations.

### 1. Scalars, Vectors, Matrices
**Explanation**: A scalar is a single number (0D). A vector is a 1D array of numbers. A matrix is a 2D grid of numbers.
**Worked Example**:
```python
import numpy as np
scalar = np.array(5)
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4]])
```
**Practice Task**: Create a 3x3 matrix containing the numbers 1 through 9.
**Common Mistakes**: Confusing Python lists of lists with NumPy arrays.
**Checkpoint**: What is the dimensionality of a vector?

### 2. Array Creation, Indexing, Slicing, Shapes
**Explanation**: Arrays are created using functions like `np.zeros`, `np.ones`, or `np.array`. Indexing accesses single elements, slicing accesses subarrays. The `.shape` attribute tells you the size of each dimension.
**Worked Example**:
```python
arr = np.zeros((2, 3)) # Shape: 2 rows, 3 columns
print(arr.shape) # (2, 3)
```
**Practice Task**: Create a 4x4 array of ones. Slice it to get the inner 2x2 square.
**Common Mistakes**: Out-of-bounds indexing. Forgetting that NumPy slicing creates views, not copies.
**Checkpoint**: If `arr.shape` is `(5, 5)`, what does `arr[1:4, 1:4]` return?

### 3. Axis Semantics and Reductions
**Explanation**: Operations like `sum` or `mean` can be applied over specific axes. `axis=0` means operation across rows (downwards), `axis=1` means across columns (rightwards).
**Worked Example**:
```python
m = np.array([[1, 2], [3, 4]])
print(np.sum(m, axis=0)) # [4, 6] (sum down the columns)
```
**Practice Task**: Find the mean of a 3x3 matrix across `axis=1`.
**Common Mistakes**: Confusing `axis=0` (collapsing the row dimension) with "summing the rows".
**Checkpoint**: If an array is `(10, 5)` and you sum over `axis=0`, what is the shape of the result?

### 4. Elementwise Ops and Broadcasting
**Explanation**: Arithmetic on arrays applies element-by-element. Broadcasting automatically expands smaller arrays to match larger ones during arithmetic.
**Worked Example**:
```python
a = np.array([1, 2, 3])
print(a * 2) # [2, 4, 6] (Broadcasting scalar)

b = np.array([[10], [20]]) # Shape (2, 1)
c = np.array([1, 2, 3])    # Shape (3,)
print(b + c) 
# Result shape (2, 3):
# [[11, 12, 13],
#  [21, 22, 23]]
```
**Practice Task**: Multiply a (3, 1) matrix by a (1, 4) matrix using element-wise multiplication. What is the resulting shape?
**Common Mistakes**: Assuming `*` is matrix multiplication (it is elementwise).
**Checkpoint**: Can a `(4, 3)` array broadcast with a `(3,)` array? (Yes, trailing dimensions match).

### 5. Dot Products, Matmul, Transposes
**Explanation**: The dot product is the sum of element-wise products of vectors. Matrix multiplication (`np.matmul` or `@`) combines matrices following linear algebra rules. Transpose swaps rows and columns (`.T`).
**Worked Example**:
```python
v1 = np.array([1, 2])
v2 = np.array([3, 4])
print(np.dot(v1, v2)) # 1*3 + 2*4 = 11

m1 = np.array([[1, 2]]) # (1, 2)
m2 = np.array([[3], [4]]) # (2, 1)
print(m1 @ m2) # [[11]], shape (1, 1)
print(m1.T) # [[1], [2]], shape (2, 1)
```
**Practice Task**: Compute the dot product of `[1, 0, -1]` and `[2, 2, 2]`.
**Common Mistakes**: Using `@` on shapes that don't align (e.g., inner dimensions don't match).
**Checkpoint**: If `A` is `(M, N)` and `B` is `(N, P)`, what is the shape of `A @ B`?

### 6. Boolean Masks and Indexing
**Explanation**: Arrays of booleans can be used to select elements from another array.
**Worked Example**:
```python
arr = np.array([1, 5, 2, 8])
mask = arr > 3 # [False, True, False, True]
print(arr[mask]) # [5, 8]
```
**Practice Task**: Extract all negative numbers from an array `np.array([4, -1, 3, -5, 0])`.
**Common Mistakes**: Using Python's `and`/`or` instead of bitwise `&`/`|` when combining masks.
**Checkpoint**: How do you set all values greater than 10 to exactly 10 in an array?

### 7. Vectorization vs Loops
**Explanation**: Vectorization pushes loops into optimized C code. You should rarely write Python `for` loops to iterate over array elements.
**Worked Example**:
```python
arr = np.arange(1000)
# Slow loop
res_loop = [x * 2 for x in arr]
# Fast vectorization
res_vec = arr * 2
```
**Practice Task**: Write a vectorized expression to compute $y = 3x + 2$ for an array `x` of 100 random numbers.
**Common Mistakes**: Reverting to `for` loops when array operations are perfectly capable of doing it instantly.
**Checkpoint**: Why is vectorization faster? (Hint: C-level loops, SIMD instructions).

### 8. Numerical Precision, Float Error, Shape Debugging
**Explanation**: Floats have limited precision. `0.1 + 0.2 != 0.3`. Shape mismatch is the #1 bug in ML.
**Worked Example**:
```python
print(0.1 + 0.2 == 0.3) # False!
print(np.isclose(0.1 + 0.2, 0.3)) # True

# Shape debugging tip: ALWAYS print shapes when things break.
```
**Practice Task**: Compare two arrays containing `[1e-10, 2e-10]` and `[0.0, 0.0]` using `np.isclose()`.
**Common Mistakes**: Checking equality of float arrays with `==`. Not checking `.shape` when a math error happens.
**Checkpoint**: What function should you use to check float equality?

### 9. Reproducible RNG
**Explanation**: Random Number Generators (RNG) need seeds to produce reproducible results.
**Worked Example**:
```python
rng = np.random.default_rng(seed=42)
print(rng.random(3)) # Will always be the same array when seeded with 42
```
**Practice Task**: Generate a 2x2 matrix of standard normal random numbers using a fixed seed.
**Common Mistakes**: Using the legacy `np.random.seed(42)` instead of the modern `default_rng`.
**Checkpoint**: Why is setting a random seed critical in the experimental workflow?

#### Self-Assessment Checklist: Module 0B
- [ ] **Skills**: Can create arrays, index, slice, apply masks, broadcast, and perform matrix multiplication.
- [ ] **Conceptual**: Explain the difference between element-wise multiplication (`*`) and matrix multiplication (`@`).
- [ ] **Hand Calculations**: Determine the resulting shape of broadcasting a `(10, 1)` array with a `(5,)` array.
- [ ] **Practice Task**: Implement a function that takes an array, standardizes it (subtracts mean, divides by standard deviation) using only vectorized operations.
- **Exit Criterion**: You can perform basic data manipulation and math entirely in NumPy without `for` loops.
- **Recovery Recommendation**: Practice NumPy slicing and broadcasting rules extensively before moving to linear algebra.

---

## <a name="module-0c-linear-algebra"></a>Module 0C: Linear Algebra

Linear algebra provides the vocabulary for representing data and parameters in machine learning.

### 1. Scalars vs Vectors vs Matrices
**Explanation**: 
- Scalar (0D): Represents magnitude. 
- Vector (1D): Represents magnitude and direction in space. A point in $n$-dimensional space.
- Matrix (2D): A collection of vectors, or a transformation.
**Worked Example**: $x = 5$ (scalar), $\mathbf{v} = [1, 2]$ (vector), $M = [[1, 2], [3, 4]]$ (matrix).
**Practice Task**: Classify the shape `(100, 10)` as scalar, vector, or matrix.
**Common Mistakes**: Treating a 1D vector (shape `(n,)`) as a 2D column matrix (shape `(n, 1)`). They behave differently in linear algebra!
**Checkpoint**: Can a vector represent a row of data?

### 2. Dimensions and Shapes
**Explanation**: The "dimension" of a vector is the number of elements it has (e.g., a 3D vector). The "dimension" of an array (NumPy) is the number of axes. A dataset of 100 samples with 5 features is a $100 \times 5$ matrix.
**Worked Example**: A dataset with `N=10` samples and `D=3` features has shape `(10, 3)`.
**Practice Task**: If you have 50 images, each $28 \times 28$ pixels, what is the shape of the flattened 2D matrix?
**Common Mistakes**: Confusing mathematical dimension (length of vector) with tensor dimension (number of axes).
**Checkpoint**: In a shape `(M, N)`, what does `M` conventionally represent in ML datasets?

### 3. Vector Addition and Scalar Multiplication
**Explanation**: Vectors add component-wise. Scalars multiply into every component. Geometrically, addition translates points, scalar multiplication scales them.
**Worked Example**: $\mathbf{u} = [1, 2], \mathbf{v} = [3, 4] \Rightarrow \mathbf{u} + \mathbf{v} = [4, 6]$. $3 \mathbf{u} = [3, 6]$.
**Practice Task**: Compute $2 \times [1, -1] + [0, 4]$ by hand.
**Common Mistakes**: Adding vectors of different mathematical dimensions (undefined).
**Checkpoint**: Geometrically, what does multiplying a vector by $-1$ do?

### 4. Dot Products and Weighted Sums
**Explanation**: The dot product measures alignment between vectors. It's the sum of element-wise products. In ML, a prediction is often a dot product of a weight vector and a feature vector (a weighted sum).
**Worked Example**: $\mathbf{w} = [0.5, 0.5]$, $\mathbf{x} = [10, 20]$. Dot product = $0.5(10) + 0.5(20) = 15$.
**Practice Task**: Calculate the dot product of $[1, 2, 3]$ and $[-1, 0, 1]$.
**Common Mistakes**: Treating dot product as returning a vector (it returns a scalar).
**Checkpoint**: If the dot product of two non-zero vectors is 0, what geometric relationship do they have? (Orthogonal).

### 5. Matrix Multiplication
**Explanation**: Matrix multiplication applies a linear transformation (a matrix) to vectors (or other matrices). Row-by-column dot products.
**Worked Example**: 
$$ \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 1x + 2y \\ 3x + 4y \end{bmatrix} $$
**Practice Task**: Multiply a $2 \times 2$ identity matrix by any $2 \times 1$ vector. What is the result?
**Common Mistakes**: Reversing the order. $AB \neq BA$. Order matters!
**Checkpoint**: To multiply an $A \times B$ matrix by a $C \times D$ matrix, what must be true about $B$ and $C$?

### 6. Transposes
**Explanation**: Flipping a matrix over its diagonal. Rows become columns.
**Worked Example**: 
$$ \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix} $$
**Practice Task**: What is the transpose of a column vector of shape `(3, 1)`?
**Common Mistakes**: Thinking transpose changes a 1D NumPy array `(3,)` to a column vector. (It doesn't change 1D arrays).
**Checkpoint**: What is $(A^T)^T$?

### 7. Norms, Distances, Squared Distances
**Explanation**: The $L2$ norm (Euclidean length) of a vector is $\sqrt{\sum x_i^2}$. The distance between two vectors is the norm of their difference.
**Worked Example**: Vector $\mathbf{v} = [3, 4]$. Norm $= \sqrt{3^2 + 4^2} = 5$.
**Practice Task**: Calculate the L2 norm of the vector $[1, 1, 1, 1]$.
**Common Mistakes**: Forgetting to square root for the exact distance, though in ML we often optimize the *squared* distance to avoid computing the square root.
**Checkpoint**: Why is squared distance often preferred in optimization over regular distance?

### 8. Linear Independence, Rank, Invertibility
**Explanation**: Vectors are linearly independent if none can be written as a combination of the others. The rank of a matrix is the number of linearly independent rows/cols. A square matrix is invertible if it has full rank.
**Worked Example**: $[1, 0]$ and $[0, 1]$ are independent. $[1, 1]$ and $[2, 2]$ are dependent.
**Practice Task**: Are the vectors $[1, 2]$ and $[-1, -2]$ independent?
**Common Mistakes**: Attempting to invert a non-square matrix directly without using a pseudo-inverse.
**Checkpoint**: If a dataset has perfectly duplicated columns (features), what happens to its rank?

### 9. Eigenvectors and Eigenvalues (Preparing for PCA)
**Explanation**: An eigenvector of a matrix is a vector that doesn't change direction when the matrix is applied to it, only stretches by a scalar (the eigenvalue). $A\mathbf{v} = \lambda\mathbf{v}$.
**Worked Example**: If applying $A$ to $\mathbf{v}$ yields $3\mathbf{v}$, then $\mathbf{v}$ is an eigenvector and 3 is the eigenvalue.
**Practice Task**: If an eigenvalue is 0, what does the matrix transformation do to the corresponding eigenvector?
**Common Mistakes**: Thinking eigenvectors can be the zero vector (they cannot).
**Checkpoint**: How do eigenvectors relate to finding the "principal components" of data?

#### Self-Assessment Checklist: Module 0C
- [ ] **Skills**: Can multiply matrices, compute dot products, and calculate L2 norms by hand and in code.
- [ ] **Conceptual**: Explain why order matters in matrix multiplication.
- [ ] **Hand Calculations**: Compute the matrix product of a $2 \times 3$ and $3 \times 2$ matrix.
- [ ] **Practice Task**: Write a function that takes two vectors and computes their Euclidean distance without using `np.linalg.norm`.
- **Exit Criterion**: You can confidently match shapes for matrix multiplication and understand geometric interpretations of vectors.
- **Recovery Recommendation**: Watch 3Blue1Brown's "Essence of Linear Algebra" series on YouTube.

---

## <a name="module-0d-calculus-and-optimization"></a>Module 0D: Calculus and Optimization

Calculus helps us understand how changing parameters affects the model's error.

### 1. Functions as I/O
**Explanation**: A mathematical function $f(x)$ takes an input and maps it to an output. In ML, the loss function $L(w)$ maps model parameters to an error value.
**Worked Example**: $f(x) = x^2$. If $x=3, f(3)=9$.
**Practice Task**: Define a function $f(w, b) = 2w + b$. Evaluate $f(3, 1)$.
**Common Mistakes**: Confusing the variables of the data ($x$) with the variables of the model ($w$). When training, $x$ is fixed data, $w$ are the variables.
**Checkpoint**: In $L(w) = (y - wx)^2$, what are we trying to minimize with respect to?

### 2. Graphs, Slopes, Rate of Change
**Explanation**: The slope of a curve at a point tells you how steep it is. A positive slope means the function is increasing; negative means decreasing.
**Worked Example**: For $f(x) = 2x$, the slope is constantly 2.
**Practice Task**: For a U-shaped curve (parabola), what is the slope at the very bottom?
**Common Mistakes**: Assuming slope is constant on non-linear curves.
**Checkpoint**: If you want to find the minimum of a function, what should you look for regarding its slope?

### 3. Derivatives with Simple Polynomials
**Explanation**: The derivative $f'(x)$ gives a formula for the slope at any point $x$. Power rule: derivative of $x^n$ is $nx^{n-1}$.
**Worked Example**: $f(x) = x^2 \Rightarrow f'(x) = 2x$. At $x=3$, slope is $2(3)=6$.
**Practice Task**: Find the derivative of $f(x) = 3x^3$.
**Common Mistakes**: Forgetting constants (derivative of $cx$ is $c$, derivative of a constant is 0).
**Checkpoint**: What is the derivative of $f(w) = w^2 + 5w + 10$?

### 4. Partial Derivatives
**Explanation**: When a function has multiple inputs, a partial derivative measures the rate of change with respect to one input, treating all others as constant.
**Worked Example**: $f(x, y) = x^2 y$. $\frac{\partial f}{\partial x} = 2xy$. $\frac{\partial f}{\partial y} = x^2$.
**Practice Task**: Find partial derivatives of $f(w_1, w_2) = w_1^2 + 3w_2$.
**Common Mistakes**: Getting confused and applying the product rule to variables that are being treated as constants.
**Checkpoint**: When computing $\frac{\partial f}{\partial y}$ for $f(x,y)$, how do you treat $x$?

### 5. Gradients as Vectors
**Explanation**: The gradient $\nabla f$ is simply a vector containing all the partial derivatives. It points in the direction of the steepest *ascent* (increase) of the function.
**Worked Example**: For $f(x, y) = x^2 + y^2$, $\nabla f = [2x, 2y]$.
**Practice Task**: Evaluate the gradient of $f(x, y) = x^2 + y^2$ at the point $(1, -2)$.
**Common Mistakes**: Thinking the gradient points towards the minimum (it points towards the maximum).
**Checkpoint**: How do we use the gradient to find the minimum? (We move in the opposite direction: $-\nabla f$).

### 6. Chain Rule
**Explanation**: Used to take derivatives of nested functions. If $y = f(u)$ and $u = g(x)$, then $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$.
**Worked Example**: $f(x) = (2x + 1)^2$. Let $u = 2x+1$. $\frac{df}{du} = 2u$. $\frac{du}{dx} = 2$. So $\frac{df}{dx} = 2u \cdot 2 = 4(2x+1)$.
**Practice Task**: Find the derivative of $(w^2 - 3)^3$ with respect to $w$.
**Common Mistakes**: Forgetting to multiply by the derivative of the inner function.
**Checkpoint**: Why is the chain rule essential in deep learning?

### 7. Loss/Objective Functions
**Explanation**: A function that quantifies "how bad" our model is doing. We want to minimize it.
**Worked Example**: Mean Squared Error (MSE): $L(w) = (prediction - actual)^2$.
**Practice Task**: If prediction is 5 and actual is 3, what is the squared error?
**Common Mistakes**: Minimizing the wrong function or maximizing a loss instead of minimizing.
**Checkpoint**: Is a lower loss always better? (Yes, for training data, but beware overfitting).

### 8. GD with One Scalar Param
**Explanation**: Gradient Descent (GD). Update rule: $w_{new} = w_{old} - \alpha \cdot \frac{dL}{dw}$, where $\alpha$ is the learning rate.
**Worked Example**: $L(w) = w^2$. $w_{old} = 4$. $dL/dw = 2w = 8$. Let $\alpha = 0.1$. $w_{new} = 4 - 0.1(8) = 3.2$. (Moving towards 0).
**Practice Task**: Perform one step of GD for $L(w) = 2w^2$ starting at $w=2$ with $\alpha = 0.1$.
**Common Mistakes**: Adding the gradient instead of subtracting it (this maximizes instead of minimizes).
**Checkpoint**: What happens if the learning rate is 0?

### 9. GD with Multiple Params
**Explanation**: We update all parameters simultaneously using the gradient vector. $\mathbf{w}_{new} = \mathbf{w}_{old} - \alpha \nabla L$.
**Worked Example**: $\mathbf{w}_{old} = [2, 3]$. $\nabla L = [4, 6]$. $\alpha = 0.1$. $\mathbf{w}_{new} = [2, 3] - [0.4, 0.6] = [1.6, 2.4]$.
**Practice Task**: Update parameter vector $[10, -5]$ with gradient $[2, -1]$ and $\alpha=0.5$.
**Common Mistakes**: Updating parameters sequentially instead of simultaneously in theory.
**Checkpoint**: Why do we subtract $\alpha \nabla L$?

### 10. Learning Rates, Convergence, Divergence
**Explanation**: The learning rate ($\alpha$) controls step size. Too small = slow convergence. Too large = divergence (overshooting the minimum and exploding).
**Worked Example**: If $\alpha$ is huge, $w$ bounces: $4 \rightarrow -10 \rightarrow 25 \rightarrow -70$.
**Practice Task**: Sketch a 1D parabola and draw arrows showing what happens if the step size is too large.
**Common Mistakes**: Assuming a fixed learning rate is always optimal.
**Checkpoint**: How can you tell if your learning rate is too high just by looking at the loss over time?

#### Self-Assessment Checklist: Module 0D
- [ ] **Skills**: Compute basic derivatives, apply chain rule, and execute a gradient descent step by hand.
- [ ] **Conceptual**: Explain what a gradient represents geometrically.
- [ ] **Hand Calculations**: Compute partial derivatives for $f(w,b) = (wx + b - y)^2$ with respect to $w$ and $b$.
- [ ] **Practice Task**: Write a Python loop that minimizes $y = x^2$ starting at $x=10$ using GD.
- **Exit Criterion**: You deeply understand that minimizing loss means calculating derivatives and taking steps in the opposite direction.
- **Recovery Recommendation**: Review single-variable derivatives and the chain rule on Khan Academy.

---

## <a name="module-0e-probability-and-statistics"></a>Module 0E: Probability and Statistics

We deal with data that is noisy. Statistics gives us tools to handle uncertainty.

### 1. Mean, Median, Variance, Standard Deviation
**Explanation**: Mean is the average. Median is the middle value. Variance measures spread (average squared distance from mean). Std Dev is the square root of variance.
**Worked Example**: Data: $[1, 2, 3, 4, 5]$. Mean=3, Median=3, Var=2, StdDev=$\sqrt{2}$.
**Practice Task**: Compute by hand the mean and variance of $[2, 4, 6]$.
**Common Mistakes**: Confusing population variance (divide by $N$) with sample variance (divide by $N-1$). In ML, we usually use $N$.
**Checkpoint**: If you add 10 to every data point, what happens to the mean? The variance?

### 2. Samples, Populations, Distributions
**Explanation**: The population is all possible data. A sample is the subset we actually observe. A distribution describes the probability of different values occurring.
**Worked Example**: We sample 100 people's heights from the population of a city. The distribution of heights often forms a bell curve.
**Practice Task**: Why do ML models use a sample instead of the whole population?
**Common Mistakes**: Assuming the sample perfectly represents the population (sampling bias).
**Checkpoint**: What is the difference between an empirical distribution (the data you have) and a theoretical distribution (like a perfect Gaussian)?

### 3. Covariance and Correlation
**Explanation**: Covariance measures how two variables vary together. Correlation is a normalized covariance between -1 and 1.
**Worked Example**: Height and weight have a positive correlation.
**Practice Task**: If $X$ and $Y$ always move in perfectly opposite directions, what is their correlation?
**Common Mistakes**: "Correlation implies causation" fallacy.
**Checkpoint**: If covariance is 0, does it mean the variables are independent? (No, only no *linear* relationship).

### 4. Conditional Probability
**Explanation**: $P(A|B)$ is the probability of event A happening given that event B has already happened.
**Worked Example**: $P(\text{rain} | \text{cloudy})$ is higher than $P(\text{rain})$.
**Practice Task**: If rolling a die, what is $P(6 | \text{even number})$?
**Common Mistakes**: Confusing $P(A|B)$ with $P(B|A)$.
**Checkpoint**: Write the formula for $P(A, B)$ using conditional probability.

### 5. Bayes' Theorem
**Explanation**: A mathematical formula to update beliefs based on new evidence. $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$.
**Worked Example**: Updating the probability a patient has a disease ($A$) given a positive test result ($B$).
**Practice Task**: Identify the "prior", "likelihood", and "posterior" in Bayes theorem.
**Common Mistakes**: Ignoring the base rate (prior probability) when calculating posterior probabilities.
**Checkpoint**: Why is the denominator $P(B)$ often difficult to compute in complex ML models?

### 6. Random Variables and Expectation
**Explanation**: A random variable maps outcomes of random events to numbers. Expectation $\mathbb{E}[X]$ is the long-run average (probability-weighted sum).
**Worked Example**: A fair coin gives $1 for heads, $0 for tails. $\mathbb{E}[X] = 0.5(1) + 0.5(0) = 0.5$.
**Practice Task**: Calculate the expectation of a single roll of a fair 6-sided die.
**Common Mistakes**: Treating expectation as an outcome that must occur (you can't roll a 3.5).
**Checkpoint**: How is the mean of a sample related to the expectation of a random variable?

### 7. Gaussian Distributions
**Explanation**: The normal (Gaussian) distribution is characterized by its mean $\mu$ and variance $\sigma^2$. It appears everywhere due to the Central Limit Theorem.
**Worked Example**: A bell curve centered at $\mu=0$ with spread $\sigma=1$ is the standard normal.
**Practice Task**: Roughly what percentage of data falls within 1 standard deviation of the mean in a Gaussian? (~68%).
**Common Mistakes**: Assuming all data is Gaussian (e.g., incomes are right-skewed).
**Checkpoint**: What shape does a Gaussian curve have?

### 8. Likelihood and Log-Likelihood
**Explanation**: Likelihood is the probability of the *observed data* given specific model parameters. We often maximize the log-likelihood (MLE) because logs turn products of probabilities into sums, which are numerically stable and easier to differentiate.
**Worked Example**: If data $X = [x_1, x_2]$, $L(\theta) = P(x_1|\theta) P(x_2|\theta)$. Log-likelihood $= \log P(x_1|\theta) + \log P(x_2|\theta)$.
**Practice Task**: Why does taking the log not change the location of the maximum? (Log is monotonic).
**Common Mistakes**: Confusing likelihood (function of parameters given data) with probability (function of data given parameters).
**Checkpoint**: Why do we often minimize *negative* log-likelihood?

### 9. Conditional Independence
**Explanation**: Variables $A$ and $B$ are conditionally independent given $C$ if knowing $C$ makes $A$ and $B$ uninformative about each other.
**Worked Example**: Shoe size and reading ability are conditionally independent given age.
**Practice Task**: Give an example of two variables that are correlated but conditionally independent given a third variable.
**Common Mistakes**: Assuming marginal independence implies conditional independence (or vice versa).
**Checkpoint**: How does Naive Bayes use conditional independence?

### 10. Intuition Behind Latent Variables
**Explanation**: Latent variables are hidden underlying factors that we don't observe directly, but which generate the data we do see.
**Worked Example**: We observe someone coughing and having a fever. The "latent variable" is whether they have the flu.
**Practice Task**: Name a possible latent variable that dictates a movie's ratings across different users. (e.g., the movie's "genre" or "quality").
**Common Mistakes**: Trying to measure latent variables directly.
**Checkpoint**: Why do clustering algorithms essentially search for a discrete latent variable?

### Prerequisite Mapping Table

| Concept | Used Extensively In |
|---|---|
| Variance/StdDev | Linear Regression, Feature Scaling, PCA |
| Likelihood/MLE | Logistic Regression, Naive Bayes, GMMs |
| Bayes Theorem | Naive Bayes |
| Expectation | EM Algorithm, GMMs |
| Covariance | PCA |

#### Self-Assessment Checklist: Module 0E
- [ ] **Skills**: Compute basic probability, mean, and variance.
- [ ] **Conceptual**: Distinguish between probability and likelihood.
- [ ] **Hand Calculations**: Apply Bayes' theorem to a simple word problem.
- [ ] **Practice Task**: Write code to compute the sample variance of a list of numbers from scratch.
- **Exit Criterion**: You understand what a Gaussian distribution is and why we sum log-probabilities instead of multiplying raw probabilities.
- **Recovery Recommendation**: Review basic probability theory and summary statistics.

---

## <a name="module-0f-machine-learning-core-concepts"></a>Module 0F: Machine Learning Core Concepts

### 1. What ML is/isn't
**Explanation**: ML is about learning rules from data to make predictions, rather than explicitly programming the rules. It is not magic; it's applied function approximation.
**Worked Example**: Traditional: `if age > 18: label = "adult"`. ML: Learn from a dataset of ages and labels to find the boundary.
**Practice Task**: Name one task easily solved by rules, and one requiring ML.
**Checkpoint**: Why use ML for spam detection instead of regular expressions?

### 2. Features, Targets, Labels, Params, Predictions
**Explanation**: 
- **Features ($X$)**: The input data (e.g., square footage).
- **Targets/Labels ($y$)**: The true answer we want to predict (e.g., house price).
- **Params ($\theta$ or $W$)**: The internal weights the model learns.
- **Predictions ($\hat{y}$)**: What the model outputs.
**Worked Example**: Features: [2000 sqft], Label: $300k. Model predicts: $290k ($\hat{y}$).
**Checkpoint**: Are parameters part of the data or part of the model?

### 3. Supervised vs Unsupervised
**Explanation**: Supervised learning has targets/labels. Unsupervised learning has only features (finding patterns/clusters).
**Worked Example**: Supervised: Predicting price from features. Unsupervised: Grouping customers by purchasing habits.
**Checkpoint**: Is grouping emails into "spam" vs "not spam" based on human tags supervised or unsupervised?

### 4. Regression vs Classification
**Explanation**: Subcategories of supervised learning. Regression predicts a continuous number. Classification predicts a discrete category.
**Worked Example**: Regression: Predicting temperature. Classification: Predicting Rain/No Rain.
**Checkpoint**: Is predicting age regression or classification? (Usually regression, though can be framed as classification).

### 5. Training Data and Inference
**Explanation**: Training is the process of adjusting parameters using data. Inference is using the frozen, trained model to make predictions on new data.
**Worked Example**: You train a model on past stock data. Inference is running it live tomorrow to guess the closing price.
**Checkpoint**: Do parameters change during inference?

### 6. Train/Val/Test Splits
**Explanation**: We split data to simulate how the model will perform in the real world. Train on training set. Tune hyperparameters on validation set. Final evaluation on test set.
**Worked Example**: Data: 1000 items. 800 train, 100 val, 100 test.
**Checkpoint**: Why can't you tune hyperparameters on the test set? (It causes data leakage / overfitting to the test set).

### 7. Baseline Models
**Explanation**: A trivial model to compare against. E.g., always predicting the mean (regression) or the majority class (classification). If your complex ML model can't beat the baseline, it's useless.
**Worked Example**: In a dataset with 90% dogs and 10% cats, the baseline always predicts "dog" and is 90% accurate.
**Checkpoint**: Why is 90% accuracy potentially terrible in medical diagnosis?

### 8. Loss Functions vs Evaluation Metrics
**Explanation**: Loss is the differentiable function the optimizer uses (e.g., MSE, Cross-Entropy). Evaluation metrics are human-readable scores (e.g., Accuracy, F1-score) used to judge performance.
**Worked Example**: Training a classifier uses Cross-Entropy Loss, but we evaluate its success using Accuracy.
**Checkpoint**: Why don't we use Accuracy as a loss function for GD? (It's not differentiable; it's a step function).

### 9. Underfitting, Overfitting, Generalization
**Explanation**: 
- Underfitting: Model is too simple, high error on train and test.
- Overfitting: Model memorized the training data, low train error, high test error.
- Generalization: Model performs well on unseen data (the goal).
**Worked Example**: Overfitting is like a student memorizing a practice test but failing the real exam.
**Checkpoint**: How does comparing train loss and val loss help detect overfitting?

### 10. Feature Scaling/Normalization
**Explanation**: ML algorithms work best when features are on a similar scale (e.g., between 0 and 1, or mean 0, variance 1). It prevents features with large numeric ranges from dominating.
**Worked Example**: Age (0-100) and Income (0-1,000,000). Without scaling, income dominates the distance calculations.
**Checkpoint**: Name one scaling technique. (Min-Max scaling, Standardization).

### 11. Data Leakage
**Explanation**: When information from outside the training dataset is used to create the model. This includes using the test set for scaling or imputation!
**Worked Example**: Computing the mean of the *entire* dataset (train+test) to standardize features before splitting.
**Checkpoint**: When standardizing data, do you compute the mean/std on the train set only, or the train+test set? (Train only!)

### 12. Reproducibility and Random Seeds
**Explanation**: ML involves randomness (initialization, shuffling). Setting a random seed ensures you get the exact same result every time you run the code.
**Worked Example**: `np.random.seed(42)`
**Checkpoint**: Why is reproducibility critical when debugging ML code?

### 13. Why Training-Only Evaluation is Insufficient
**Explanation**: A model can perfectly memorize training data (e.g., a lookup table). Evaluating on training data tells you nothing about its predictive power.
**Worked Example**: A 1-Nearest Neighbor algorithm has 0 training loss but might fail miserably on test data.
**Checkpoint**: What is the ultimate goal of machine learning? (Generalization to unseen data).

#### Self-Assessment Checklist: Module 0F
- [ ] **Skills**: Can articulate the ML problem setup, recognize data leakage, and set up baselines.
- [ ] **Conceptual**: Differentiate between parameters and hyperparameters.
- [ ] **Hand Calculations**: Compute baseline accuracy for an imbalanced dataset (95% Class A, 5% Class B).
- [ ] **Practice Task**: Write a snippet to split a NumPy array into 80% train and 20% test manually.
- **Exit Criterion**: You deeply grasp why we split data and the difference between regression and classification.
- **Recovery Recommendation**: Read the introduction of "Introduction to Statistical Learning" (ISLR).

---

## <a name="the-experimental-workflow"></a>The Experimental Workflow

When implementing algorithms from scratch, you will hit bugs. ML bugs are silent—the code runs, but the math is wrong, leading to poor learning. To survive, you must act like a scientist, not just a programmer.

**Debugging vs Testing vs Experimentation:**
- Debugging fixes syntax/crashes.
- Testing verifies isolated logic.
- Experimentation is tracking how changes affect learning metrics.

### The 10-Step Apprenticeship Workflow

For every algorithm you build, follow this exact loop:

1. **Hypothesize**: "I believe my linear regression update rule is correct."
2. **Choose dataset**: Use a toy dataset that is hand-computable. E.g., $X = [[1], [2], [3]]$, $y = [2, 4, 6]$.
3. **Identify baseline**: If I predict mean $y=4$, MSE is $8/3$.
4. **Predict expected behavior**: "With weights initialized to 0, first prediction is 0. Loss will be high. One step of GD with $\alpha=0.1$ should make weights positive."
5. **Implement minimum**: Write only the forward pass and one update step.
6. **Run experiment**: Run the single step.
7. **Compare against expectation**: Did the weights become positive exactly as hand-calculated?
8. **Inspect failures**: If weights became negative, check the gradient sign.
9. **Change one factor**: Fix the sign. Do not change learning rate and the formula at the same time.
10. **Record observation**: Keep notes on how initialization or learning rate affected the specific algorithm.

**Note-Keeping**: Write down when a model explodes due to learning rate vs when it just learns slowly. This builds intuition.

---

## <a name="stage-0-exit-self-assessment"></a>Stage 0 Exit Self-Assessment

Before moving to Chapter 1, verify you can answer these definitively:

**Conceptual Questions:**
1. What is the difference between a list and a NumPy array?
2. Why is vectorization faster than `for` loops in Python?
3. Geometrically, what does the gradient vector represent?
4. Why do we need a test set?
5. What does the learning rate control in gradient descent?

**Hand Calculations:**
1. $X = [[1, 2], [3, 4]]$. Compute $X \cdot [1, -1]^T$.
2. Given $L(w) = (w - 3)^2$, calculate $w$ after one step of GD starting at $w=0$, $\alpha=0.5$.
3. Compute the sample variance of $[0, 4, 8]$.

**Final Practice Task:**
Write a Python class `DummyRegressor` that takes a `learning_rate` in `__init__`, takes a NumPy array $X$ and $y$ in `fit`, loops 10 times, and updates a single weight scalar by subtracting `learning_rate * mean(X * y)`. (This isn't real linear regression, but it tests the API and NumPy skills).

**Explicit Exit Criterion:**
If you can complete the conceptual questions, hand calculations, and the final practice task without relying heavily on Google, you are ready for Chapter 1: Linear Regression.

**Recovery Guidance:**
If you stumbled on the math, review Modules 0C/0D. If you stumbled on NumPy syntax, review Module 0B. If the `class` structure was confusing, review Module 0A.

---
*Ready? Proceed to Chapter 1!*
