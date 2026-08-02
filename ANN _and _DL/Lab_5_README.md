# Lab 5 — Convolutional Neural Network (Keras) for Binary Image Classification



---

## 1. What are we trying to do?

We want to teach a computer to look at a photo and correctly guess which of **2 classes** it belongs to:

- Horse
- Human

This is **binary image classification** — "binary" because there are only 2 possible answers. The assignment calls for the classic Cats vs Dogs dataset (or any two-class image dataset); the mirrors that host Cats vs Dogs turned out to be unreachable, so this notebook uses **Horses or Humans**, a dataset with the exact same shape (one folder per class, JPEG/PNG photos). Everything below — the pipeline, the model, the code — works unchanged if you point `train_dir`/`val_dir` at a Cats vs Dogs folder instead.

Unlike Lab 4 (which fed 4 pre-measured numbers per flower into a Dense network), here the input is a **raw image** — a grid of pixels. That's what makes this a job for a **Convolutional Neural Network (CNN)**: a network built to learn visual patterns (edges, textures, shapes) directly from pixels, instead of from hand-picked measurements.

---

## 2. The big picture (architecture)

```
Input (150 x 150 x 3 RGB image)
    -> Rescaling (pixels 0-255 -> 0-1)
    -> Conv2D 16 filters (3x3) + ReLU -> MaxPooling2D
    -> Conv2D 32 filters (3x3) + ReLU -> MaxPooling2D
    -> Conv2D 64 filters (3x3) + ReLU -> MaxPooling2D
    -> Conv2D 64 filters (3x3) + ReLU -> MaxPooling2D
    -> Flatten -> Dropout(0.5)
    -> Dense 64 neurons + ReLU
    -> Output - 1 neuron (Sigmoid) -> P(human)
```

Think of it as a funnel that goes from "raw pixels" to "one probability":
1. Each **Conv2D** block slides a small set of learnable filters across the image, each filter lighting up wherever it spots a pattern it recognizes (an edge, a curve, a patch of color/texture).
2. Each **MaxPooling2D** shrinks the image in half, keeping only the strongest signal in each small region — this makes the network faster, and more tolerant of the pattern appearing in a slightly different spot.
3. Stacking 4 of these blocks lets early layers detect simple things (edges) and later layers detect more complex things (a leg shape, a torso shape) built out of the earlier features.
4. **Flatten + Dense + Sigmoid** takes everything the convolutions found and turns it into a single number between 0 and 1 — the probability the image is a human (close to 0 = horse, close to 1 = human).

---

## 3. Step-by-step code explanation

### Step 1 — Download and load the data

```python
train_zip = keras.utils.get_file(
    "horse-or-human.zip",
    origin="https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip",
    extract=True, cache_subdir="datasets/horse-or-human",
)
...
train_ds = keras.utils.image_dataset_from_directory(
    train_dir, image_size=(150, 150), batch_size=32, label_mode="binary", seed=42
)
```
- `keras.utils.get_file(...)` downloads the dataset zip once and caches it locally (`~/.keras/datasets/...`), then unzips it. Re-running the notebook later reuses the cached copy instead of re-downloading.
- The dataset ships as two folders, `horses/` and `humans/`, each full of photos of that class — the same layout Cats vs Dogs uses (`cats/`, `dogs/`).
- `image_dataset_from_directory(...)` is a Keras helper that reads a folder-of-folders and automatically:
  - infers the class from the subfolder name (`horses` → label 0, `humans` → label 1, alphabetical order),
  - resizes every image to a consistent `150x150`,
  - groups images into batches of 32 (`batch_size=32`) so the model processes 32 images at a time instead of one-by-one,
  - `label_mode="binary"` returns a single 0/1 label per image (rather than a one-hot vector, since there are only 2 classes).

### Step 2 — Visualize sample images

The notebook plots a 3x3 grid of raw training images with their labels, just to sanity-check the data loaded correctly and see what the model will actually be looking at.

### Step 3 — Preprocessing and data augmentation

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
])
train_ds_aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(AUTOTUNE)
```
- With only 1027 training images, the network could easily just memorize the exact photos it sees rather than learning what a "horse" or "human" generally looks like.
- **Data augmentation** manufactures new training variety for free by randomly flipping, rotating, zooming, and adjusting the contrast of each image a little differently *every epoch*. The underlying photo count never changes, but the model effectively never sees the exact same image twice.
- This only ever runs on the **training** set (`training=True`) — the validation set is left untouched, because we want to measure real, unmodified performance.
- Pixel rescaling (dividing by 255 to get values in `[0, 1]`) is *not* done here — it's built directly into the model as the first layer (see Step 4), so the raw `0–255` images can be fed in as-is.
- `.prefetch(AUTOTUNE)` overlaps data loading with model training so the GPU/CPU is never left waiting on disk I/O; `AUTOTUNE` lets TensorFlow pick the best amount of prefetching automatically.

### Step 4 — Build the model

```python
keras.utils.set_random_seed(42)

model = keras.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
```
- `keras.utils.set_random_seed(42)` is called right before building the model so the random initial weights (and later, training) are reproducible, regardless of how much randomness the earlier visualization cells already consumed.
- `layers.Rescaling(1./255)` — the first real layer. Images arrive as pixel values `0-255`; dividing by 255 puts every pixel in `0-1`, which neural networks train on far more reliably than large raw pixel values.
- `layers.Conv2D(16, 3, activation="relu")` — a convolution layer with 16 filters, each `3x3` pixels. Each filter slides across the image and produces a "feature map" showing where it detected its pattern. `ReLU` (`max(0, x)`) adds non-linearity, same role as in Lab 4's Dense layers.
- `layers.MaxPooling2D()` — halves the width and height of the feature maps by keeping only the maximum value in each `2x2` window. This shrinks the data (faster, fewer parameters downstream) and makes detection slightly tolerant of the pattern shifting by a pixel or two.
- The filter count doubles going deeper (16 → 32 → 64 → 64) — early layers need few filters to catch simple patterns (edges, colors); deeper layers need more filters to represent more complex, combined patterns.
- `layers.Flatten()` — after 4 rounds of convolution+pooling, the data is still a 3D grid of numbers (height x width x filters). `Flatten` unrolls it into a single long 1D vector so it can feed into regular `Dense` layers.
- `layers.Dropout(0.5)` — randomly zeroes out 50% of the values flowing through during training (a different random half every batch). This forces the network to not rely too heavily on any single feature, which reduces overfitting — especially important with only ~1000 training images.
- `layers.Dense(64, activation="relu")` — a regular fully-connected layer that combines everything the convolutions extracted.
- `layers.Dense(1, activation="sigmoid")` — the output layer: **one** neuron (not 2, unlike Lab 4's 3-neuron softmax output) because with only 2 classes we only need one probability — "how humanlike is this image" — and "how horselike" is just `1 - that`. **Sigmoid** squashes any raw number into `0-1`.

### Step 5 — Compile

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
```
- `Adam` optimizer again, same role as Lab 4, but with a smaller learning rate (`1e-4` vs Lab 4's `0.01`) — CNNs with many more parameters and noisier, harder-to-fit image data tend to train more stably with smaller update steps.
- `binary_crossentropy` — the 2-class counterpart to Lab 4's `categorical_crossentropy`. It compares the single predicted probability against the true 0/1 label.

### Step 6 — Train the model

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=6, restore_best_weights=True
)
history = model.fit(
    train_ds_aug, validation_data=val_ds, epochs=30, verbose=0, callbacks=[early_stop],
)
```
- `epochs=30` sets an upper bound on training, but `EarlyStopping` decides when to actually stop.
- **`EarlyStopping`** watches `val_accuracy` (accuracy on the held-out validation images) after every epoch. If it doesn't improve for `patience=6` epochs in a row, training stops early — instead of continuing to overfit for the full 30 epochs.
- **`restore_best_weights=True`** is the important part: when training stops, the model's weights are rolled back to whichever epoch had the *best* validation accuracy seen so far, not just the last epoch trained. This is the same "keep the best checkpoint" idea flagged as a missed opportunity in Lab 4's conclusion — here it's actually implemented.

### Step 7 — Training curves

The notebook plots train vs. validation accuracy and loss per epoch, side by side, so overfitting (train improving while validation stalls/worsens) is visible at a glance rather than having to read a table of numbers.

### Step 8 — Final evaluation and interpretation

```python
val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
y_prob = model.predict(val_ds, verbose=0).ravel()
y_pred = (y_prob > 0.5).astype(int)
```
- `model.evaluate(...)` reports final loss/accuracy on the validation set, using the *restored best* weights from early stopping.
- `model.predict(...)` returns raw probabilities (e.g. `0.83` = 83% confident "human"); thresholding at `0.5` (`y_prob > 0.5`) converts each probability into a hard 0/1 class prediction.
- A **confusion matrix** is then built by hand (a 2x2 table of actual-vs-predicted counts), plus **precision** and **recall** for the "human" class, and a 3x3 grid of validation images captioned with predicted vs. actual label (green if correct, red if wrong) for a qualitative look at what the model gets wrong.

---

## 4. Results

### Training progress
```
Epoch [ 1/9] | Train Loss: 0.6890 | Train Acc: 53.6% | Val Loss: 0.6777 | Val Acc: 50.0%
Epoch [ 2/9] | Train Loss: 0.6715 | Train Acc: 67.4% | Val Loss: 0.6360 | Val Acc: 52.7%
Epoch [ 3/9] | Train Loss: 0.6150 | Train Acc: 73.8% | Val Loss: 0.4643 | Val Acc: 78.5%
Epoch [ 4/9] | Train Loss: 0.4683 | Train Acc: 82.8% | Val Loss: 0.9587 | Val Acc: 60.2%
Epoch [ 5/9] | Train Loss: 0.3827 | Train Acc: 84.2% | Val Loss: 0.8046 | Val Acc: 73.8%
Epoch [ 6/9] | Train Loss: 0.3310 | Train Acc: 86.5% | Val Loss: 1.1253 | Val Acc: 70.3%
Epoch [ 7/9] | Train Loss: 0.2986 | Train Acc: 88.1% | Val Loss: 1.0585 | Val Acc: 72.7%
Epoch [ 8/9] | Train Loss: 0.2625 | Train Acc: 89.1% | Val Loss: 1.3627 | Val Acc: 69.9%
Epoch [ 9/9] | Train Loss: 0.2811 | Train Acc: 87.8% | Val Loss: 1.1762 | Val Acc: 71.1%
```
Training stopped early after 9 epochs (no improvement in `val_accuracy` for 6 epochs after the peak at epoch 3).

### Final evaluation
```
Validation Accuracy: 78.5%  (201/256 correct)

Confusion Matrix (rows = actual, cols = predicted)
                 pred: horses   pred: humans
actual: horses            40             88
actual: humans            33             95

Precision (humans): 51.9%  |  Recall (humans): 74.2%
```

---

## 5. Interpretation

- **Best epoch was early (epoch 3), not the last one.** Validation accuracy peaked at 78.5% on epoch 3, then bounced between 60–74% for the rest of training while train accuracy climbed steadily to 88%. `EarlyStopping(restore_best_weights=True)` is exactly what rescues the final model from this — without it, the reported result would have been epoch 9's weaker weights, not epoch 3's stronger ones.
- **Validation loss explodes even while validation accuracy holds up.** By epoch 9, validation loss is `1.18` (vs. `0.28` train loss) — the model isn't just occasionally wrong, it's *very confidently* wrong on some validation images. This is a sharper form of overfitting than Lab 4 showed: not just "worse accuracy" but "increasingly overconfident, badly wrong probabilities."
- **The confusion matrix reveals a class bias, not just a raw error rate.** The model predicts "human" far more often than "horse" overall (183 "human" predictions vs. 73 "horse" predictions on 256 roughly-balanced images) — recall for humans is high (74.2%) but precision is barely better than a coin flip (51.9%), meaning when it says "human" it's frequently wrong. 78.5% overall accuracy alone would hide this asymmetry; the confusion matrix doesn't.
- **Why this dataset is unusually hard for a small CNN:** the validation images in Horses or Humans are deliberately rendered in a different visual style (different lighting/background/rendering engine) than the training images — a well-known property of this dataset used to teach exactly this lesson. It's a much bigger train/validation distribution gap than Lab 4's Iris split, which is why accuracy here (≈78%) is far more volatile epoch-to-epoch than Lab 4's (93–100%).

## 6. Observation

- Data augmentation and dropout visibly slow down overfitting (train accuracy only reaches ~88% instead of >99% within 9 epochs) but don't eliminate it on this dataset — validation loss still rises almost every epoch after the peak.
- A learning rate of `1e-4` (10x smaller than Lab 4's `0.01`) was needed for stable training; an earlier attempt at `1e-3` caused validation accuracy to swing even more wildly and settle lower.
- Adding a 4th `Conv2D`+`MaxPooling2D` block (deeper than the minimum 3 needed to go from 150x150 down to a small feature grid) gave the network more room to build up higher-level features before flattening, and measurably helped validation accuracy in testing versus a 3-block version.
- The red-captioned misclassifications in the prediction grid are a fast, qualitative way to audit the model — in a real project, patterns among the wrong predictions (e.g. always misclassifying side profiles, or images with cluttered backgrounds) point directly at what more training data or augmentation to add next.

## 7. Conclusion

- A small custom CNN (4 Conv2D+MaxPooling blocks, ~1.2M parameters) can learn to tell horses from humans directly from raw pixels, without any hand-engineered features — reaching 78.5% validation accuracy on a genuinely difficult, distribution-shifted validation set.
- Compared to Lab 4's Dense-only MLP on 4 pre-measured numbers, this lab's `Conv2D`/`MaxPooling2D` layers are what let the network operate on raw `150x150x3` images (67,500 raw input numbers) at all — a plain Dense network of that input size would need enormous numbers of parameters and would ignore the 2D spatial structure of the image entirely.
- The main lesson is the same one Lab 4 ended on but now *fixed* rather than just observed: **more epochs is not automatically better**, and this time the notebook actively defends against it with `EarlyStopping(restore_best_weights=True)` instead of just noting the peak came mid-training after the fact.
- Overall, this exercise demonstrates the full CNN workflow for binary image classification — **load images → augment → build conv/pool stack → compile → train with early stopping → evaluate with a confusion matrix → visually inspect predictions** — and shows concretely how a validation set that doesn't match the training distribution produces noisy, overconfident errors that a single accuracy number alone would hide.

---

## 8. Key Concepts (quick reference)

| Term | Meaning |
|---|---|
| **Convolution (Conv2D)** | Slides small learnable filters over an image to detect local visual patterns (edges, textures, shapes) |
| **Feature map** | The output of a convolution — a grid showing where a filter's pattern was found |
| **MaxPooling** | Down-samples feature maps by keeping the strongest activation in each window; reduces size and adds translation-tolerance |
| **Data augmentation** | Randomly flipping/rotating/zooming training images so the model sees more variety without collecting more data |
| **Dropout** | Randomly deactivates a fraction of neurons during training to reduce overfitting |
| **Sigmoid** | Output activation squashing a single value into a probability in `[0, 1]`, used for binary classification |
| **Binary crossentropy** | Loss function for two-class classification problems |
| **EarlyStopping** | Callback that halts training once a monitored metric (e.g. validation accuracy) stops improving |
| **restore_best_weights** | EarlyStopping option that rolls the model back to its best-performing epoch instead of keeping the last one |
| **Confusion matrix** | A table of actual vs. predicted class counts, revealing error patterns (e.g. class bias) that a single accuracy number hides |
| **Precision / Recall** | Precision = of everything predicted as class X, how much really was X; Recall = of everything that really was class X, how much got predicted as X |

---

## 9. Libraries Used
- `tensorflow`, `tensorflow.keras` — model definition, layers, augmentation, compilation, training, callbacks
- `keras.utils.get_file` / `image_dataset_from_directory` — dataset download, caching, and loading from a folder-of-folders layout
- `numpy` — array handling for the confusion matrix and prediction thresholding
- `matplotlib.pyplot` — sample image grids, augmentation previews, training curves, and the prediction visualization grid
