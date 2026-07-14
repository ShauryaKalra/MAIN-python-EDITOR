# Lab 2 — Single Neuron Model (Manual Implementation)

## Objective
Build a **single artificial neuron** from scratch using NumPy — without any deep learning library. Train it using **gradient descent** and test it on the AND gate.

---

## Difference from Lab 1 (Perceptron)
| Feature | Lab 1 Perceptron | Lab 2 Single Neuron |
|---|---|---|
| Activation | Step function (hard 0/1) | Sigmoid (smooth 0–1 probability) |
| Learning rule | Perceptron update rule | Gradient descent |
| Output | Binary (0 or 1) | Probability (e.g. 0.944) |
| Loss function | None (error count) | Binary cross-entropy |
| Update style | Per sample | Batch (all samples at once) |

---

## Code Breakdown

### Function: `sigmoid(z)`
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
The **activation function** used by this neuron. Unlike the step function in Lab 1, sigmoid outputs a smooth probability between 0 and 1:
- Large positive `z` → output close to `1`
- Large negative `z` → output close to `0`
- `z = 0` → output exactly `0.5`

This makes it differentiable, which allows gradient descent to work.

---

### Class: `Neuron`

#### `__init__(n_inputs, lr, epochs)`
```python
def __init__(self, n_inputs, lr=0.1, epochs=1000):
    self.w  = np.zeros(n_inputs)
    self.b  = 0.0
    self.lr = lr
    self.epochs = epochs
```
| Parameter | Description |
|---|---|
| `n_inputs` | Number of input features (2 for AND gate) |
| `lr` | Learning rate — controls step size during gradient descent |
| `epochs` | Number of times to loop through the full dataset |
| `self.w` | Weight array, one per input, initialized to zeros |
| `self.b` | Bias, initialized to 0 |

---

#### Method: `predict(X)`
```python
def predict(self, X):
    z = np.dot(X, self.w) + self.b
    return sigmoid(z)
```
- `np.dot(X, self.w)` — weighted sum of all inputs
- `+ self.b` — adds the bias
- `sigmoid(z)` — converts the sum into a probability between 0 and 1

---

#### Method: `train(X, y)`
```python
def train(self, X, y):
    m = len(y)
    for epoch in range(self.epochs):
        y_hat = self.predict(X)
        error = y_hat - y

        self.w -= self.lr * (1/m) * np.dot(X.T, error)
        self.b -= self.lr * (1/m) * np.sum(error)
```
This is **batch gradient descent**:
1. `y_hat = self.predict(X)` — get predictions for all samples at once
2. `error = y_hat - y` — compute how wrong each prediction is
3. `np.dot(X.T, error)` — compute gradient of the loss w.r.t. weights
4. `np.sum(error)` — compute gradient w.r.t. bias
5. Subtract scaled gradient from weights and bias to reduce loss
6. Every 200 epochs, prints the **binary cross-entropy loss**

**Loss formula (Binary Cross-Entropy):**
```
Loss = -mean( y * log(y_hat) + (1 - y) * log(1 - y_hat) )
```
Lower loss = better predictions. The `1e-8` added inside `log()` prevents `log(0)` errors.

---

## Dataset — AND Gate
```python
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 0, 0, 1], dtype=float)
```
Only outputs `1` when **both** inputs are `1`. All other combinations output `0`.

---

## Training Output
```
Epoch  200 | Loss: 0.1436
Epoch  400 | Loss: 0.0816
Epoch  600 | Loss: 0.0565
Epoch  800 | Loss: 0.0431
Epoch 1000 | Loss: 0.0347
```
Loss steadily decreases — the neuron is learning.

---

## Final Results
```
Final weights: [6.014  6.014]
Final bias   : -9.198

Input [0 0] → prob=0.000  predicted=0  actual=0  ✓
Input [0 1] → prob=0.040  predicted=0  actual=0  ✓
Input [1 0] → prob=0.040  predicted=0  actual=0  ✓
Input [1 1] → prob=0.944  predicted=1  actual=1  ✓
```
- Both weights are equal (~6) because both inputs contribute equally to AND
- The bias is strongly negative (-9.2) to ensure the neuron only fires when both inputs are 1
- Prediction threshold: `prob >= 0.5` → predicted `1`, else `0`

---

## Key Concepts
| Term | Meaning |
|---|---|
| **Sigmoid** | Smooth activation that outputs a probability between 0 and 1 |
| **Gradient Descent** | Optimization algorithm that nudges weights in the direction that reduces loss |
| **Learning Rate** | How large each gradient descent step is (too high = overshoots, too low = slow) |
| **Batch Gradient Descent** | Updates weights using all samples at once (vs. one at a time in Lab 1) |
| **Binary Cross-Entropy** | Standard loss function for binary classification problems |
| **Bias** | Shifts the decision boundary so the neuron isn't forced to pass through the origin |
| **Epoch** | One complete pass through the entire training dataset |

---

## Libraries Used
- `numpy` — array math, dot products, exp, log
