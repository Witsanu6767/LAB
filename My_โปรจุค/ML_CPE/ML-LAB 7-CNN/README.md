# 🕷️ CNN Image Classification: Spider vs Centipede 🐛

An end-to-end Computer Vision pipeline to classify image data of **Spiders** and **Centipedes** using **Convolutional Neural Networks (CNN)** built with TensorFlow/Keras. The project covers data loading, preprocessing, stratified dataset splitting, model training with anti-overfitting callbacks, evaluation, and inference testing on real samples.

---

## 📌 System Architecture

The workflow is divided into 6 modular stages:

```
[ PetImages Data ] 
       │
       ▼
 1. data_loader.py       ──► Loads BGR images and handles initial resizing
       │
       ▼
 2. preprocessing.py   ──► Converts BGR to RGB & formats to NumPy Arrays (uint8)
       │
       ▼
 3. split_data.py        ──► Splits Train / Validation / Test sets using Stratified Sampling
       │
       ▼
 4. cnn_model.py         ──► Trains CNN with Data Augmentation, Dropout, and Callbacks
       │
       ▼
 5. evaluate.py          ──► Calculates Accuracy, Confusion Matrix & plots Training History
       │
       ▼
 6. test_cnn.py          ──► Performs random inference testing on the Test Set & outputs visual predictions
```

---

## 📁 Project Structure

```
lab7/
├── PetImages/                    # Dataset directory
│   ├── Spider/                   # Spider image folder
│   └── Centipede/                # Centipede image folder
├── classification/               # Source code directory
│   ├── data_loader.py            # Loads images and filters out corrupted files
│   ├── preprocessing.py         # Converts Color Space (BGR->RGB) and formats data
│   ├── split_data.py            # Stratified dataset split script
│   ├── cnn_model.py             # CNN Architecture and Training pipeline
│   ├── evaluate.py              # Evaluates metrics and plots performance charts
│   ├── main.py                  # Main script to execute the entire pipeline
│   ├── test_cnn.py              # Inference script for random test predictions
│   └── outputs/                 # Artifacts and saved model outputs
│       ├── cnn_model.keras      # Saved trained model (Best Weights)
│       ├── classes.json         # Class index mappings
│       ├── history.json         # Loss and Accuracy metrics per epoch
│       ├── training_history.png # Training history curves
│       ├── confusion_matrix.png # Confusion matrix plot
│       └── prediction_sample.png# Random sample inference output image
└── README.md
```

---

## 🛠️ CNN Architecture (`cnn_model.py`)

Designed for optimal performance while preventing **Overfitting** on small-to-medium datasets:

* **In-Model Normalization:** `Rescaling(1./255)` layer scales pixel values ($0–255 \rightarrow 0–1$) directly inside the model.
* **Data Augmentation:** Includes `RandomFlip("horizontal")` and `RandomRotation(0.05)` to increase training variance.
* **Feature Extraction (3 Blocks):**
  * **Block 1:** Conv2D (32 Filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)
  * **Block 2:** Conv2D (64 Filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)
  * **Block 3:** Conv2D (128 Filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)
* **Classification Head:** 
  * `Flatten()` + `Dense(128, activation='relu')`
  * `Dropout(0.4)` to randomly drop neurons and mitigate memorization.
  * `Dense(num_classes, activation='softmax')` for multi-class probability outputs.
* **Optimization & Callbacks:**
  * **Optimizer:** Adam ($\text{Learning Rate} = 10^{-4}$)
  * **Loss Function:** Sparse Categorical Crossentropy
  * **EarlyStopping:** Halts training when validation loss stops improving.
  * **ReduceLROnPlateau:** Automatically decays learning rate when loss plateaus.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install tensorflow opencv-python scikit-learn matplotlib numpy
```

### 2. Dataset Setup
Organize your images inside the `PetImages/` directory structured by class name:
```
PetImages/
├── Spider/
└── Centipede/
```

### 3. Run Training & Evaluation Pipeline
Execute the full pipeline from data processing to evaluation with a single command:
```bash
python classification/main.py
```

### 4. Run Inference Test
Test the saved model on random samples from the test set:
```bash
python classification/test_cnn.py
```

---

## 📊 Artifacts & Outputs

All output files are saved to `classification/outputs/` upon successful execution:

| File Name | Type | Description |
| :--- | :--- | :--- |
| **`training_history.png`** | Image | Comparison plots for Loss & Accuracy across Epochs (Train vs Val) |
| **`confusion_matrix.png`** | Image | Matrix showing correct vs incorrect prediction distribution per class |
| **`prediction_sample.png`** | Image | Visual prediction results with confidence scores (%) and correctness status |
| **`cnn_model.keras`** | Model File | Exported Keras model ready for deployment |
