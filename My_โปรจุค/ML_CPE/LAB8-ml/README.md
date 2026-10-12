# 🕷️ VGG Image Recognition Project: Spider vs. Centipede (PyTorch & OpenCV)

This project implements a Deep Learning image classification pipeline using a **VGG-style** architecture powered by **PyTorch** and **OpenCV**, designed specifically for multi-class/binary image recognition tasks such as classifying **Spiders vs. Centipedes**.

## 📂 Project Structure & File Overview

The codebase follows a modular design pattern to ensure maintainability and scalability:

| File | Primary Role | 
| ----- | ----- | 
| **`main.py`** | Main entry point coordinating the complete pipeline from data loading to evaluation. | 
| **`data_loader.py`** | Automatically detects classes from subdirectories, loads images, and filters out corrupted files. | 
| **`preprocessing.py`** | Converts image color space from BGR to RGB and resizes images efficiently using `cv2.INTER_AREA`. | 
| **`split_data.py`** | Splits the dataset into Train, Validation, and Test sets using stratified sampling to preserve class balance. | 
| **`vgg_model.py`** | Implements the `PyTorchVGG` architecture, handles training/inference loops, and supports GPU/CPU execution. | 
| **`evaluate.py`** | Computes evaluation metrics (Accuracy, Classification Report, Confusion Matrix) and generates performance plots. | 
| **`test_vgg.py`** | Randomly samples images from the test set to run inference and visualize prediction confidence. | 

## ⚙️ Pipeline Workflow

1. **📥 Loading Data:** Scans subdirectories (e.g., `Spider` and `Centipede`), loads valid image files, and limits the count per class if specified.

2. **🎨 Preprocessing:** Resizes and converts images to contiguous NumPy arrays ready for tensor conversion.

3. **✂️ Splitting:** Divides the dataset into training, validation, and testing subsets using stratified sampling.

4. **🚀 Training:** Trains the PyTorch VGG model using the Adam optimizer and saves the best model weights to `outputs/vgg_model.pt`.

5. **📊 Evaluation & Testing:** Evaluates model performance, saving the confusion matrix and training history graphs into the `outputs/` directory.

## 🚀 Getting Started

### 1. Dataset Preparation

Organize your dataset folder structure (e.g., `PetImages` or `InsectsDataset` located in the parent directory) with subfolders named after each target class:

```
Dataset/
├── Spider/
│   ├── 001.jpg
│   └── 002.jpg
└── Centipede/
    ├── 001.jpg
    └── 002.jpg
```

### 2. Training the Model

Review and adjust hyperparameters like `IMG_SIZE`, `EPOCHS`, or `BATCH_SIZE` inside `main.py`, then run:

```bash
python main.py
```

### 3. Testing and Inference

After training completes, run inference on random test samples to visualize predicted labels (`Spider` vs. `Centipede`) and model confidence:

```bash
python test_vgg.py
```

*💡 All output logs, trained model weights (`vgg_model.pt`), and generated visualization plots will be saved automatically inside the `outputs/` folder.*
