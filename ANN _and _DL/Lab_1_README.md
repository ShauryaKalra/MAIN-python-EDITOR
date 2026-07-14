# Lab 1 — Perceptron Simulation using NumPy

## Objective
Simulate a **Perceptron** — the simplest form of an artificial neuron — from scratch using only NumPy, and test it on logic gates (AND, OR, NAND, XOR).

---

## What is a Perceptron?
A Perceptron is a single neuron that:
1. Takes multiple inputs
2. Multiplies each input by a weight and adds a bias
3. Passes the result through a **step function**
4. Outputs either `0` or `1`

**Weight update rule (Perceptron Learning Rule):**
```
w = w + lr * (actual - predicted) * x
b = b + lr * (actual - predicted)
```
Weights only change when the prediction is wrong. When all predictions are correct, training stops early (converged).

---

## Code Breakdown

### Class: `Perceptron`
Defined in Cell 2. Encapsulates all the logic of a single perceptron.

| Component | Description |
|---|---|
| `__init__(learning_rate, epochs)` | Sets the learning rate (how big each weight update step is) and max number of training passes |
| `self.weights` | NumPy array of one weight per input feature, initialized to zeros |
| `self.bias` | A single bias value, initialized to 0 |

---

### Method: `step_function(z)`
```python
def step_function(self, z):
    return 1 if z >= 0 else 0
```
The **activation function** of a perceptron. Takes the weighted sum `z` and returns:
- `1` if `z >= 0`
- `0` if `z < 0`

This is a hard threshold — either fires or doesn't.

---

### Method: `predict(x)`
```python
def predict(self, x):
    z = np.dot(x, self.weights) + self.bias
    return self.step_function(z)
```
- `np.dot(x, self.weights)` — computes the weighted sum of inputs
- Adds the bias
- Passes through `step_function` to get a 0 or 1 output

---

### Method: `train(X, y)`
```python
def train(self, X, y):
    ...
    for epoch in range(self.epochs):
        for xi, yi in zip(X, y):
            y_pred = self.predict(xi)
            error = yi - y_pred
            self.weights += self.lr * error * xi
            self.bias    += self.lr * error
```
- Loops over all training samples for each epoch
- Computes prediction and error for every sample
- Updates weights and bias using the Perceptron Learning Rule
- Prints epoch number, current weights, bias, and number of errors
- Stops early if `total_error == 0` (all samples predicted correctly)

---

## Experiments

### Example 1 — AND Gate
| Input A | Input B | Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

Converges in **4 epochs**. The perceptron successfully learns AND.

---

### Example 2 — OR Gate
| Input A | Input B | Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

Converges in **4 epochs**. The perceptron successfully learns OR.

---

### Example 3 — NAND Gate
| Input A | Input B | Output |
|---|---|---|
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Converges in **6 epochs**. Weights go negative since NAND is the inverse of AND.

---

### Example 4 — XOR Gate (Fails on Purpose)
| Input A | Input B | Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Does NOT converge.** XOR is not linearly separable — no single straight line can divide the outputs into 0s and 1s. This is a fundamental limitation of the perceptron, and it motivated the invention of **multi-layer networks (MLPs)**.

---

## Key Concepts
| Term | Meaning |
|---|---|
| **Epoch** | One full pass through all training samples |
| **Learning Rate (lr)** | Controls how much weights change per update |
| **Bias** | Allows the decision boundary to shift, not just rotate |
| **Linear Separability** | A requirement for perceptron convergence — data must be separable by a straight line |
| **Convergence** | When all training samples are predicted correctly and training stops |

---

## Libraries Used
- `numpy` — for array operations and dot product
