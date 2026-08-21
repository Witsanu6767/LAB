# Image Classification using Fully-Connected Neural Network (MLP)

This repository contains an end-to-end image classification pipeline built with **TensorFlow / Keras**. The project utilizes a **Multilayer Perceptron (MLP)** architecture with memory-optimized data handling, automated image preprocessing via **OpenCV**, and comprehensive model evaluation using **scikit-learn** and **Matplotlib**.

---

## 📌 Key Features

* **Automatic Class Detection:** Automatically infers class labels directly from dataset subdirectories.
* **Memory Optimization:** Resizes images on-the-fly during loading and keeps image arrays as `uint8`. Normalization is embedded inside the network using a `Rescaling(1.0/255)` layer to conserve RAM.
* **Dynamic Model Architecture:** Automatically configures the output layer and loss function based on the detected number of classes (binary vs. multiclass).
* **Smart Training Callbacks:** Implements `EarlyStopping` to prevent overfitting and `ReduceLROnPlateau` to dynamically adjust the learning rate during training.
* **Comprehensive Evaluation:** Automatically generates accuracy/loss curves, a confusion matrix visualization, and a 2x2 prediction grid on random test samples.

---

## 📁 Project Structure

```text
.
├── AlienImages/             # Raw dataset directory (organized by subdirectories)
├── main.py                  # Main execution pipeline (Load -> Train -> Evaluate)
├── data_loader.py           # Loads and resizes images from disk
├── preprocessing.py         # Color space conversion (BGR to RGB) and array formatting
├── split_data.py            # Stratified train / validation / test dataset splitter
├── nn_model.py              # MLP model architecture, training loop, and inference
├── evaluate.py              # Generates classification reports, confusion matrix, and training curves
├── test_nn.py               # Visualizes predictions on 4 random test images in a 2x2 grid
└── outputs/                 # Directory where models, datasets (.npy), and evaluation plots are saved
```

---

## 🛠️ Model Architecture

The core model is built using `keras.Sequential` with the following structure:

1. **`Rescaling(1/255)`**: Embeds 0–1 normalization directly into the model.
2. **`Flatten`**: Flattens input image tensors into 1D feature vectors.
3. **`Dense(256)` + `BatchNormalization` + `Dropout(0.4)`**
4. **`Dense(128)` + `BatchNormalization` + `Dropout(0.4)`**
5. **`Dense(64)` + `Dropout(0.3)`**
6. **`Output Layer`**:
   * **Binary (2 classes):** `Dense(1, activation='sigmoid')` paired with `binary_crossentropy` loss.
   * **Multiclass (>2 classes):** `Dense(num_classes, activation='softmax')` paired with `sparse_categorical_crossentropy` loss.

---

## 🚀 Getting Started

### 1. Prerequisites

Install the required Python dependencies:

```bash
pip install tensorflow opencv-python matplotlib scikit-learn numpy
```

### 2. Dataset Setup

Organize your images inside the `AlienImages/` directory structured by class folders:

```text
AlienImages/
├── Alien/
│   ├── image1.jpg
│   └── image2.jpg
└── Predator/
    ├── image1.jpg
    └── image2.jpg
```

### 3. Training the Model

Execute the primary pipeline script:

```bash
python main.py
```

Upon completion, all generated artifacts will be stored in the `outputs/` directory:
* `nn_model.keras`: Saved trained Keras model
* `training_history.png`: Accuracy and loss curves across training epochs
* `confusion_matrix.png`: Confusion matrix plot
* `history.json` & `.npy`: Dataset splits and metric history

### 4. Testing & Inference

To evaluate the trained model on 4 random images from the test set:

```bash
python test_nn.py
```

The output grid with predictions and confidence scores will be saved to `outputs/prediction_sample.png`.

---

## 📊 Evaluation & Outputs

The training process outputs standard metrics to the console and generates visual charts:
* **Classification Report:** Detailed breakdown of Precision, Recall, and F1-Score per class.
* **Loss & Accuracy Plots:** Used to monitor training convergence and detect overfitting.
* **Prediction Grid:** Displays random test samples labeled with predicted class, ground truth, and prediction confidence.
