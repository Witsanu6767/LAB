# 👽 SVM Image Recognition: Alien vs Predator 🛸

An image classification system built to distinguish between **Alien** and **Predator** using a **Support Vector Machine (SVM)** model combined with **Principal Component Analysis (PCA)** for dimensionality reduction. Developed in Python using `scikit-learn`.

---

## 📌 Key Features

- **Automated Preprocessing:** Reads images, converts them to grayscale, and resizes them to 100x100 pixels.
- **Dimensionality Reduction via PCA:** Reduces feature dimensions from 10,000 pixels down to 150 components for faster and more accurate SVM performance.
- **Classification with RBF Kernel SVM:** Uses a high-performance Support Vector Classifier to separate classes.
- **Stratified Dataset Splitting:** Splits data into Train (80%) and Test (20%) sets while preserving class proportions.
- **Visual Evaluation:** Generates a Confusion Matrix image and tests random samples with visual outputs (Green border = Correct, Red border = Incorrect).

---

## 📁 Project Structure

```
.
├── AlienImages/             # Dataset folder (each subfolder represents a class name)
│   ├── alien/               # Alien images
│   └── predator/            # Predator images
├── outputs/                 # Output folder for models and results (auto-generated)
│   ├── svm_model.pkl        # Trained SVM model
│   ├── scaler.pkl           # StandardScaler + PCA Pipeline
│   ├── confusion_matrix.png # Confusion matrix plot
│   └── ...                  # Initial raw data and scaled arrays (.npy, .json)
├── data_load.py             # Loads images from directory, resizes, and converts to grayscale
├── preprocess.py           # Normalizes pixels (0-255 -> 0-1) and flattens into a matrix
├── split_data.py            # Splits dataset into Train/Test sets (Stratified)
├── svm_model.py             # Builds and trains pipeline (StandardScaler + PCA + SVM)
├── evaluate.py              # Evaluates performance (Accuracy, Classification Report, Confusion Matrix)
├── test_model.py            # Tests random images and saves output visual sample
└── main.py                  # Main script to run the full workflow pipeline
```

---

## 🔄 Pipeline Workflow

```
[1. Load Data] ➡️ [2. Preprocess] ➡️ [3. Split Data] ➡️ [4. Train SVM+PCA] ➡️ [5. Predict & Evaluate]
```

1. **Step 1: Load Data (`data_load.py`)** — Automatically detects class folders, loads image files, converts them to Grayscale, and resizes them to 100x100.
2. **Step 2: Preprocess (`preprocess.py`)** — Flattens 2D images (100x100) into 1D vectors (10,000) and scales pixel values to 0.0 – 1.0.
3. **Step 3: Split Dataset (`split_data.py`)** — Splits data into 80% Train and 20% Test using Stratified Split.
4. **Step 4: Train Model (`svm_model.py`)** — Combines `StandardScaler` + `PCA (150 components)` + `SVC (RBF Kernel, C=10)` and saves the model.
5. **Step 5 & 6: Prediction & Evaluation (`evaluate.py`)** — Measures Accuracy, F1-Score, and exports `confusion_matrix.png`.

---

## 🚀 Getting Started

### 1. Install Required Libraries

```bash
pip install numpy opencv-python scikit-learn matplotlib joblib
```

### 2. Dataset Setup

Organize images inside the `AlienImages/` folder as follows:

```text
AlienImages/
├── alien/
│   ├── image1.jpg
│   └── image2.jpg
└── predator/
    ├── image1.jpg
    └── image2.jpg
```

### 3. Run Training Pipeline

Run the command below to start the entire process:

```bash
python main.py
```

### 4. Test Sample Predictions

After training, run this script to randomly sample test images and save an output visual example (`outputs/prediction_sample.png`):

```bash
python test_model.py
```
