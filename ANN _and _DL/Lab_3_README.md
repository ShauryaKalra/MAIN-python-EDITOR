# Lab 3 — Feedforward Neural Network (PyTorch)

## Objective
Build a **multi-layer feedforward neural network** using PyTorch (instead of manual NumPy code) to classify Iris flowers into 3 species — Setosa, Versicolor, Virginica.

---

## Difference from Lab 1 & Lab 2
| Feature | Lab 1 Perceptron | Lab 2 Single Neuron | Lab 3 Feedforward NN |
|---|---|---|---|
| Layers | 1 | 1 | 3 (2 hidden + output) |
| Framework | NumPy (manual) | NumPy (manual) | PyTorch (autograd) |
| Activation | Step function | Sigmoid | ReLU (hidden), raw logits (output) |
| Task | Binary (AND gate) | Binary (AND gate) | Multi-class (3 flower species) |
| Loss function | None (error count) | Binary cross-entropy | CrossEntropyLoss |
| Optimizer | Perceptron update rule | Manual gradient descent | Adam |
| Gradients | N/A | Computed by hand | Computed automatically (`loss.backward()`) |

---

## Network Architecture
```
Input (4 features: sepal length/width, petal length/width)
    -> Hidden Layer 1 - 16 neurons + ReLU
    -> Hidden Layer 2 - 8 neurons + ReLU
    -> Output - 3 classes (Setosa, Versicolor, Virginica)
```

---

## Code Breakdown

### 1. Load and Prepare Data
```python
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- `load_iris()` — loads the classic 150-sample, 4-feature, 3-class dataset.
- `StandardScaler` — normalizes features to zero mean / unit variance so all 4 features are on a comparable scale (important for gradient-based training).
- `train_test_split` — 80% train (120 samples) / 20% test (30 samples).
- Data is then converted to `FloatTensor` (features) / `LongTensor` (integer class labels) since PyTorch requires tensors, not NumPy arrays.

---

### 2. Define the Network
```python
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)
```
- `nn.Linear(in, out)` — a fully connected layer that learns weights + bias for the transformation.
- `nn.ReLU()` — activation applied between layers; introduces non-linearity by zeroing out negative values.
- No activation after the last `nn.Linear(8, 3)` — the raw logits are passed straight to `CrossEntropyLoss`, which applies softmax internally.
- `nn.Sequential` chains the layers so `forward()` is just one call.

---

### 3. Setup
```python
model     = FeedforwardNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```
- `CrossEntropyLoss` — standard loss for multi-class classification; combines softmax + negative log-likelihood.
- `Adam` — adaptive optimizer that adjusts the learning rate per parameter, generally converging faster than plain gradient descent.

---

### 4. Train the Model
```python
for epoch in range(epochs):
    model.train()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
This is the standard PyTorch training loop:
1. `model(X_train)` — forward pass, produces logits for all 120 training samples.
2. `criterion(predictions, y_train)` — computes the loss.
3. `optimizer.zero_grad()` — clears gradients from the previous step (PyTorch accumulates them by default).
4. `loss.backward()` — backpropagation; computes gradients of the loss w.r.t. every weight automatically (autograd).
5. `optimizer.step()` — updates all weights using those gradients.

Every 10 epochs, the model is switched to `.eval()` mode and evaluated on the held-out test set (inside `torch.no_grad()` so no gradients are tracked during evaluation).

---

## Training Output
```
Epoch [ 10/100] | Train Loss: 1.0000 | Test Loss: 0.9667 | Test Accuracy: 83.3%
Epoch [ 20/100] | Train Loss: 0.6694 | Test Loss: 0.5954 | Test Accuracy: 76.7%
Epoch [ 30/100] | Train Loss: 0.4191 | Test Loss: 0.3651 | Test Accuracy: 80.0%
Epoch [ 40/100] | Train Loss: 0.3225 | Test Loss: 0.2799 | Test Accuracy: 90.0%
Epoch [ 50/100] | Train Loss: 0.2249 | Test Loss: 0.1746 | Test Accuracy: 100.0%
Epoch [ 60/100] | Train Loss: 0.1323 | Test Loss: 0.0919 | Test Accuracy: 100.0%
Epoch [ 70/100] | Train Loss: 0.0812 | Test Loss: 0.0584 | Test Accuracy: 100.0%
Epoch [ 80/100] | Train Loss: 0.0616 | Test Loss: 0.0343 | Test Accuracy: 100.0%
Epoch [ 90/100] | Train Loss: 0.0534 | Test Loss: 0.0222 | Test Accuracy: 100.0%
Epoch [100/100] | Train Loss: 0.0488 | Test Loss: 0.0205 | Test Accuracy: 100.0%
```
Both train and test loss decrease steadily, and test accuracy climbs to 100% by epoch 50 and stays there — no overfitting signs since test loss keeps falling alongside train loss.

---

## Final Results
```
Test Accuracy: 100.0%  (30/30 correct)

Sample Predictions vs Actual:
  Predicted: versicolor   | Actual: versicolor
  Predicted: setosa       | Actual: setosa
  Predicted: virginica    | Actual: virginica
  Predicted: versicolor   | Actual: versicolor
  Predicted: versicolor   | Actual: versicolor
```
All 30 test samples classified correctly. Iris is a relatively easy, well-separated dataset, so a small 2-hidden-layer network reaches perfect accuracy quickly.

---

## Key Concepts
| Term | Meaning |
|---|---|
| **Feedforward** | Data flows in one direction only — input → hidden → output, no loops |
| **ReLU** | Activation function that outputs `max(0, x)`; adds non-linearity, replaces negatives with 0 |
| **CrossEntropyLoss** | Loss function for multi-class classification; internally applies softmax + log-likelihood |
| **Adam Optimizer** | Adaptive gradient-based optimizer that adjusts each weight's learning rate individually |
| **Backpropagation** | Algorithm that computes gradients by propagating the error backwards through the network |
| **Autograd** | PyTorch's automatic differentiation engine — computes gradients without manual derivative code |
| **Epoch** | One complete pass through the entire training dataset |
| **Standardization** | Scaling features to zero mean / unit variance so no single feature dominates due to scale |

---

## Libraries Used
- `torch`, `torch.nn`, `torch.optim` — model definition, layers, loss, optimizer
- `sklearn.datasets` — Iris dataset
- `sklearn.model_selection` — train/test split
- `sklearn.preprocessing` — feature standardization
