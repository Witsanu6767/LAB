# ML-06: Neural Network (NN) Image Recognition

An end-to-end image recognition pipeline (Alien vs Predator) using a **Multi-Layer Perceptron (MLP)** / **Neural Network (NN)** built with a modular Python structure.

---

## 📌 Project Overview

This repository implements a complete Machine Learning pipeline for Image Classification using a Fully-Connected Neural Network (MLP). It covers everything from raw image loading and preprocessing to dataset splitting, model training with Keras/TensorFlow, evaluation, and sample inference.

---

## 📁 Project Structure

```text
.
├── data_loader.py       # Automatically loads images, detects classes, and resizes inputs
├── preprocessing.py     # Handles color conversions (BGR to RGB) and formats features as uint8
├── split_data.py        # Splits data into Train / Validation / Test sets
├── nn_model.py          # Defines the MLP architecture, training, and inference functions
├── evaluate.py          # Calculates accuracy metrics, classification reports, and saves plots
├── main.py              # Main driver script executing Steps 1 to 6
├── test_nn.py           # Inference test script that predicts on random samples with confidence scores
└── outputs/             # Directory containing saved models (.keras), arrays (.npy), and plots
```

---

## ⚙️ Pipeline Workflow

The entire workflow is orchestrated by `main.py` across 6 key steps:

### 1. Data Loading (`data_loader.py`)
- Automatically detects class labels from subdirectories (`Alien`, `Predator`).
- Resizes all incoming images to a unified $100 \times 100$ pixel resolution.
- Saves `labels.npy` and `classes.json` inside the `outputs/` folder.

### 2. Preprocessing (`preprocessing.py`)
- Converts OpenCV's default **BGR** color space to **RGB** for proper visualization and analysis.
- Retains data in `uint8` format for memory efficiency (deferring the $0-255$ normalization to an in-model layer).

### 3. Dataset Splitting (`split_data.py`)
- Employs a **Stratified Split** to preserve target class distribution across splits.
- Data distribution ratio:
  - **Test Set**: 20%
  - **Validation Set**: 10% (carved from training data to monitor overfitting)
  - **Training Set**: 70%

### 4. Neural Network Training (`nn_model.py`)
- **Model Architecture (MLP)**:
  - `Rescaling(1.0 / 255)`: Rescales pixel values from $0-255$ to $0-1$ inside the network.
  - `Flatten`: Flattens 2D images ($100 \times 100 \times 3$) into a 1D vector (30,000 features).
  - `Dense Layers`: Hidden layers of size 256, 128, and 64 equipped with `BatchNormalization` and `Dropout` (0.3 - 0.4) to combat overfitting.
  - `Output Layer`: Single-node Dense layer using a `Sigmoid` activation for binary classification (trained via `binary_crossentropy` loss).
- **Optimization & Callbacks**:
  - **Adam Optimizer** ($lr = 10^{-3}$).
  - `EarlyStopping`: Halts training when `val_loss` stops improving for 5 consecutive epochs, restoring the best weights.
  - `ReduceLROnPlateau`: Dynamically reduces learning rate when progress plateaus.
- Saves the trained model to `outputs/nn_model.keras`.

### 5. Prediction (`nn_model.py`)
- Runs inference on the test set (`X_test`).
- Calculates output probabilities and maps them to binary class labels using a $0.5$ threshold.

### 6. Evaluation (`evaluate.py`)
- Calculates and logs **Accuracy**, **Classification Report** (Precision, Recall, F1-Score), and the **Confusion Matrix**.
- Exports evaluation visual plots to `outputs/`:
  - `confusion_matrix.png`: Visual representation of true vs predicted labels.
  - `training_history.png`: Epoch-by-epoch loss and accuracy curves.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install numpy opencv-python matplotlib scikit-learn tensorflow
```

### 2. Execute Full Pipeline (Train & Evaluate)
```bash
python main.py
```

### 3. Run Sample Inference Test
Randomly selects 4 test images and produces visual predictions alongside percentage confidence scores:
```bash
python test_nn.py
```

---

## 📊 Artifacts Generated in `outputs/`

After running the project, the following files will be produced:
- **`nn_model.keras`**: Saved Keras model binary.
- **`classes.json` / `labels.npy` / `features.npy`**: Metadata and extracted raw features.
- **`X_train.npy`, `X_val.npy`, `X_test.npy`, `y_train.npy`, `y_val.npy`, `y_test.npy`**: Pre-split numpy arrays.
- **`history.json`**: Full training history log.
- **`confusion_matrix.png`**: Confusion matrix visualization.
- **`training_history.png`**: Training vs Validation Accuracy and Loss graph.
- **`prediction_sample.png`**: $2 \times 2$ grid plot showing sample predictions with confidence levels.
