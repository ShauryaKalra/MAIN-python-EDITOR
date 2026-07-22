# Lab 4 — Multi-Layer Perceptron (Keras) for Multiclass Classification

A beginner-friendly, line-by-line walkthrough of `Lab_4.ipynb`. No prior Keras experience assumed — every new function is explained the first time it shows up.

---

## 1. What are we trying to do?

We want to teach a computer to look at 4 measurements of a flower (sepal length, sepal width, petal length, petal width) and correctly guess which of **3 species** it is:

- Setosa
- Versicolor
- Virginica

This is called **multiclass classification** — "multi" because there are more than 2 possible answers (unlike Lab 1/2, which only predicted yes/no for an AND gate).

We solve it by building a small neural network called a **Multi-Layer Perceptron (MLP)**: a stack of layers where every neuron in one layer is connected to every neuron in the next layer (also called "fully connected" or "Dense" layers).

---

## 2. The big picture (architecture)

```
Input (4 numbers: the 4 flower measurements)
    -> Dense Layer 1 — 16 neurons, ReLU activation
    -> Dense Layer 2 — 8 neurons, ReLU activation
    -> Output Layer  — 3 neurons, Softmax activation (one score per species)
```

Think of it like a funnel of decision-making:
1. The **input** is the raw flower data.
2. The **hidden layers** (16 then 8 neurons) combine and re-combine the 4 numbers in more and more abstract ways, learning patterns like "petals this long usually mean Virginica."
3. The **output layer** turns everything the network learned into 3 probabilities — one per species — that add up to 100%.

---

## 3. Step-by-step code explanation

### Step 1 — Import the tools we need

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

tf.random.set_seed(42)
np.random.seed(42)
```
- `tensorflow` / `keras` — the deep learning library we use to build and train the network. Keras is a friendly, high-level interface built into TensorFlow.
- `sklearn` (scikit-learn) — used only for the boring-but-important data prep (loading the dataset, splitting it, scaling it).
- `set_seed(42)` — neural networks start with **random** weights and shuffle data randomly. Fixing the "seed" makes those random choices repeatable, so you get the same result every time you re-run the notebook. It doesn't change *how* the network learns, just makes it reproducible.

### Step 2 — Load the data

```python
iris = load_iris()
X = iris.data        # the 4 measurements, shape (150, 4)
y = iris.target       # the species label, 0/1/2, shape (150,)
class_names = iris.target_names
```
- `load_iris()` fetches the classic Iris dataset: 150 flowers, 4 measurements each, one of 3 species.
- `X` (capital, by convention) holds the **inputs/features** — what the model sees.
- `y` (lowercase) holds the **targets/labels** — what the model is supposed to predict, stored as plain integers (0 = Setosa, 1 = Versicolor, 2 = Virginica).

### Step 3 — Scale the features

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```
- Petal length might range 1–7 cm while sepal width ranges 2–4 cm. Without scaling, the network could pay more attention to whichever feature happens to have bigger numbers, not whichever is actually more useful.
- `StandardScaler` rewrites every feature so it has **mean 0** and **standard deviation 1** ("standardizing"). This puts all 4 features on equal footing and helps the network train faster and more reliably.
- `fit_transform` does two things at once: `fit` calculates the mean/std of each feature, `transform` applies the scaling.

### Step 4 — One-hot encode the labels

```python
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
```
- The labels start as plain numbers: `0`, `1`, `2`. But the network's output layer produces **3 separate probabilities** (one per species), not a single number. To compare "3 probabilities" against "the correct answer," the correct answer also needs to be in that 3-number shape.
- **One-hot encoding** turns each label into a vector with a single `1` in the position of the correct class and `0`s everywhere else:
  - Setosa (0)     → `[1, 0, 0]`
  - Versicolor (1) → `[0, 1, 0]`
  - Virginica (2)  → `[0, 0, 1]`
- `y.reshape(-1, 1)` just reshapes the 1D list of labels into a column so the encoder can process it.

### Step 5 — Split into train and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)
```
- We never want to test the model on data it already memorized during training — that would be like grading a student using the exact questions they were given the answers to. So we hold back 20% of the flowers as a **test set** that the model never sees during training.
- `test_size=0.2` → 30 flowers for testing, 120 for training.
- `stratify=y` → makes sure both the train and test sets have a fair, proportional mix of all 3 species (instead of, say, the test set accidentally getting mostly Setosas).

### Step 6 — Build the model

```python
model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(3, activation="softmax")
])
```
- `keras.Sequential([...])` — the simplest way to build a network in Keras: just list the layers in order, data flows straight through them top to bottom.
- `layers.Input(shape=(4,))` — tells the model to expect 4 numbers per flower.
- `layers.Dense(16, activation="relu")` — the first hidden layer: 16 neurons, each fully connected to all 4 inputs. Each neuron computes a weighted sum of the inputs, then passes it through **ReLU**.
  - **ReLU** (Rectified Linear Unit) = `max(0, x)`. It just zeroes out negative values and keeps positive ones unchanged. Without an activation function like this, stacking layers would be mathematically pointless (multiple linear layers collapse into one linear layer). ReLU adds the "bend" that lets the network learn non-straight-line (non-linear) patterns.
- `layers.Dense(8, activation="relu")` — a second hidden layer that further combines what the first layer learned, compressing 16 values down to 8.
- `layers.Dense(3, activation="softmax")` — the output layer: 3 neurons, one per species.
  - **Softmax** converts 3 raw numbers ("logits") into 3 probabilities that all add up to 1 (100%). For example, raw scores `[2.0, 0.5, 0.1]` might become probabilities `[0.75, 0.17, 0.08]`. Whichever number is highest is the model's predicted species.

**Why 16 and 8 neurons?** There's no strict formula — these are small, reasonable choices for a tiny dataset (120 training rows, 4 features). Fewer neurons risk under-fitting (network too simple to learn the pattern); way more risks over-fitting (network memorizes rather than generalizes) on such a small dataset.

### Step 7 — Compile the model

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```
Compiling tells Keras *how* the model should learn:
- **`optimizer`** — the algorithm that adjusts the network's weights after every batch of data, to make future predictions less wrong. `Adam` is a popular optimizer that automatically adapts how big each weight update is, generally converging faster and more reliably than plain gradient descent. `learning_rate=0.01` controls how big each adjustment step is — too high and training becomes unstable, too low and training is very slow.
- **`loss`** — the function that measures "how wrong" a prediction is. `categorical_crossentropy` compares the model's predicted probability distribution (e.g. `[0.75, 0.17, 0.08]`) against the true one-hot label (e.g. `[1, 0, 0]`) and produces a single number — smaller means the prediction is closer to correct. This is the number the optimizer is trying to minimize.
- **`metrics=["accuracy"]`** — an extra number we track just to make sense of progress as humans (% of predictions that were exactly right). It is *not* used for training itself — only `loss` drives the weight updates.

### Step 8 — Train the model

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=8,
    verbose=0
)
```
- `model.fit(...)` is where the actual learning happens. Internally, for every batch of data, Keras:
  1. Runs the batch through the network to get predictions (**forward pass**).
  2. Compares predictions to the true labels using the loss function.
  3. Computes how much each weight contributed to the error (**backpropagation**).
  4. Nudges every weight slightly to reduce the error (**optimizer step**).
- **`epochs=100`** — one epoch = one full pass through all 120 training flowers. We repeat this 100 times so the network gets many chances to refine its weights.
- **`batch_size=8`** — instead of processing all 120 training samples at once, the data is split into chunks of 8. The model updates its weights after each chunk, so in one epoch it makes 120/8 = 15 weight updates rather than just 1. This is faster and often trains better than using the whole dataset per step.
- **`validation_data=(X_test, y_test)`** — after every epoch, Keras also checks performance on the *held-out* test set (without training on it), so we can watch whether the model is genuinely learning or just memorizing the training data.
- `history` stores the loss/accuracy recorded at every epoch, for both training and validation, which we print out every 10 epochs afterward.

### Step 9 — Evaluate the final model

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
predictions = model.predict(X_test, verbose=0)
```
- `model.evaluate(...)` runs the finished model on the test set one more time and reports the final loss and accuracy.
- `model.predict(...)` returns the raw probability vector (e.g. `[0.02, 0.91, 0.07]`) for each test flower. We then use `np.argmax(...)` to pick the class with the highest probability as the model's final answer, and compare it to the true label to print sample predictions.

---

## 4. Results

### Training progress
```
Epoch [ 10/100] | Train Loss: 0.0917 | Test Loss: 0.1107 | Test Accuracy: 96.7%
Epoch [ 20/100] | Train Loss: 0.0432 | Test Loss: 0.0824 | Test Accuracy: 96.7%
Epoch [ 30/100] | Train Loss: 0.0355 | Test Loss: 0.0846 | Test Accuracy: 96.7%
Epoch [ 40/100] | Train Loss: 0.0296 | Test Loss: 0.0866 | Test Accuracy: 96.7%
Epoch [ 50/100] | Train Loss: 0.0238 | Test Loss: 0.0875 | Test Accuracy: 96.7%
Epoch [ 60/100] | Train Loss: 0.0399 | Test Loss: 0.0622 | Test Accuracy: 100.0%
Epoch [ 70/100] | Train Loss: 0.0195 | Test Loss: 0.1113 | Test Accuracy: 93.3%
Epoch [ 80/100] | Train Loss: 0.0168 | Test Loss: 0.1142 | Test Accuracy: 93.3%
Epoch [ 90/100] | Train Loss: 0.0063 | Test Loss: 0.1088 | Test Accuracy: 96.7%
Epoch [100/100] | Train Loss: 0.0047 | Test Loss: 0.1588 | Test Accuracy: 93.3%
```

### Final evaluation
```
Test Accuracy: 93.3%  (28/30 correct)

Sample Predictions vs Actual:
  Predicted: setosa       | Actual: setosa
  Predicted: virginica    | Actual: virginica
  Predicted: versicolor   | Actual: versicolor
  Predicted: versicolor   | Actual: versicolor
  Predicted: setosa       | Actual: setosa
```

---

## 5. Interpretation

- **Train loss vs. test loss**: Train loss falls almost the entire time, ending near `0.0047` — essentially perfect on training data. Test loss, however, hits its lowest point around epoch 60 (`0.0622`) and then *rises again* by epoch 100 (`0.1588`), even though train loss kept falling.
- This gap between "still improving on training data" and "getting worse on unseen data" is the textbook signature of **overfitting**: with only 243 trainable parameters and just 120 training examples, the network has more than enough capacity to start memorizing quirks of the specific training flowers instead of learning the general pattern that separates the species.
- **Accuracy is noisy, not smooth**: test accuracy bounces between 93.3% and 100% rather than climbing steadily. With only 30 test samples, each single misclassified flower changes accuracy by a full 3.3 percentage point step — small test sets naturally produce jumpy metrics.

## 6. Observation

- The model reaches strong accuracy (96–100%) very early (by epoch 10) — Iris is a well-known "easy" dataset where the 3 species are largely separable using just these 4 measurements.
- The best test performance (100% accuracy, lowest test loss) occurred around epoch 60, *before* training finished — if we had stopped training there (a technique called **early stopping**), we'd have kept the best-generalizing version of the model instead of training all the way to epoch 100.
- The two flowers the final model got wrong were most likely Versicolor/Virginica confusions — these two species are known to have overlapping petal/sepal measurements, unlike Setosa which is clearly separated from the other two in the raw data.
- Softmax output probabilities give more information than a plain right/wrong label — e.g., we could inspect *how confident* the model was on the two flowers it got wrong, which is useful for diagnosing borderline cases.

## 7. Conclusion

- A small Keras MLP (16 → 8 → 3 neurons) is more than capable of solving the Iris multiclass classification problem, reaching 93.3–100% test accuracy depending on the training epoch.
- Compared to writing the training loop by hand (as in Lab 3's PyTorch version), Keras's `model.fit(...)` collapses the forward pass, loss computation, backpropagation, and optimizer step into a single line — at the cost of having less visibility into what happens each epoch unless you inspect `history`.
- The main lesson from the training curve is that **more epochs is not automatically better**: the model's ability to generalize peaked mid-training and then degraded slightly as it began overfitting. In a real project, this is exactly the kind of situation that calls for early stopping, a validation-based checkpoint (saving the best epoch's weights), or regularization (e.g. dropout, L2 weight decay) to keep the network from memorizing the training set.
- Overall, the exercise demonstrates the standard Keras workflow — **prepare data → build model → compile → fit → evaluate** — and shows, very concretely, what overfitting looks like in a real training curve rather than just as an abstract definition.

---

## 8. Key Concepts (quick reference)

| Term | Meaning |
|---|---|
| **MLP** | A feedforward network made of one or more `Dense` (fully connected) hidden layers |
| **Dense layer** | A layer where every neuron is connected to every neuron in the previous layer |
| **ReLU** | Activation function `max(0, x)`; adds non-linearity so the network can learn curved/complex patterns |
| **Softmax** | Output activation that converts raw scores into a probability distribution over classes (all outputs sum to 1) |
| **One-hot encoding** | Representing a class label as a binary vector, e.g. class 1 of 3 → `[0, 1, 0]` |
| **Categorical Crossentropy** | Loss function for multi-class classification that compares predicted probabilities to one-hot targets |
| **Adam Optimizer** | Adaptive gradient-based optimizer that adjusts each weight's learning rate individually |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch size** | Number of samples processed before the model's weights are updated once |
| **Overfitting** | When train loss keeps falling but validation loss stops improving or rises — the model is memorizing rather than generalizing |
| **Early stopping** | Halting training once validation performance stops improving, to avoid overfitting |

---

## 9. Libraries Used
- `tensorflow`, `tensorflow.keras` — model definition, layers, compilation, training
- `sklearn.datasets` — Iris dataset
- `sklearn.model_selection` — train/test split
- `sklearn.preprocessing` — feature standardization and one-hot label encoding
- `numpy` — array handling and picking the highest-probability class (`argmax`)
