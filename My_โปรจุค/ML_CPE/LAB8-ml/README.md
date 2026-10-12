# 🐾 VGG Image Recognition Project (PyTorch & OpenCV)

This project implements a Deep Learning image classification pipeline using a **VGG-style** architecture powered by **PyTorch** and **OpenCV**, designed for tasks such as Cat vs. Dog image recognition.

## 📂 Project Structure & File Overview

The codebase follows a modular design pattern to ensure maintainability and scalability:

| File | Primary Role |
| :--- | :--- |
| **`main.py`** | Main entry point coordinating the complete pipeline from data loading to evaluation. |
| **`data_loader.py`** | Automatically detects classes from subdirectories, loads images, and filters out corrupted files. |
| **`preprocessing.py`** | Converts image color space from BGR to RGB and resizes images efficiently using `cv2.INTER_AREA`. |
| **`split_data.py`** | Splits the dataset into Train, Validation, and Test sets using stratified sampling to preserve class balance. |
| **`vgg_model.py`** | Implements the `PyTorchVGG` architecture, handles training/inference loops, and supports GPU/CPU execution. |
| **`evaluate.py`** | Computes evaluation metrics (Accuracy, Classification Report, Confusion Matrix) and generates performance plots. |
| **`test_vgg.py`** | Randomly samples images from the test set to run inference and visualize prediction confidence. |

## ⚙️ Pipeline Workflow

1. **📥 Loading Data:** Scans the directory structure, loads valid image files, and limits the count per class if specified.
2. **🎨 Preprocessing:** Resizes and converts images to contiguous NumPy arrays ready for tensor conversion.
3. **✂️ Splitting:** Divides the data into training, validation, and test subsets.
4. **🚀 Training:** Trains the PyTorch VGG model using Adam optimizer and saves the best model weights to `outputs/vgg_model.pt`.
5. **📊 Evaluation & Testing:** Evaluates model performance, saving the confusion matrix and training loss/accuracy graphs into the `outputs/` directory.

## 🚀 Getting Started

### 1. Dataset Preparation

Organize your dataset folder structure (e.g., `PetImages` located in the parent directory) with subfolders for each class:

```text
PetImages/
├── Cat/
│   ├── 001.jpg
│   └── 002.jpg
└── Dog/
    ├── 001.jpg
    └── 002.jpg
```

### 2. Training the Model

Review and adjust hyperparameters like `IMG_SIZE`, `EPOCHS`, or `BATCH_SIZE` inside `main.py`, then run:

```bash
python main.py
```

### 3. Testing and Inference

After training completes, you can test random samples from your test dataset and generate visual prediction outputs using:

```bash
python test_vgg.py
```

*💡 All output logs, trained weights, and generated plots will be saved automatically inside the `outputs/` folder.*
