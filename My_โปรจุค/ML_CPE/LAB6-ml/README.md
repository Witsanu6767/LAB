# Machine Learning Lab 6: Neural Network and Applications
---

## 📋 Project Structure

The project is modularized for easy maintenance and scalability:

* 📁 `AlienImages/` - Directory storing image datasets for training and testing (organized by sub-class folders).
* 📁 `outputs/` - Directory for storing execution results, such as trained models, confusion matrix plots, training history, and prediction samples.
* 📄 `data_loader.py` - Module for loading image datasets from directories and automatically resizing images [cite: 6].
* 📄 `preprocessing.py` - Module for handling image preprocessing (e.g., converting color space from BGR to RGB and arranging data arrays) [cite: 3].
* 📄 `split_data.py` - Module for splitting the dataset into Train, Validation, and Test sets using stratified sampling [cite: 4].
* 📄 `nn_model.py` - Neural Network model architecture (Keras Sequential), training function with callbacks, and prediction function [cite: 2].
* 📄 `evaluate.py` - Module for evaluating performance (Accuracy, Classification Report, Confusion Matrix) and plotting Loss/Accuracy curves [cite: 7].
* 📄 `main.py` - Main pipeline script to execute the entire workflow from start to finish [cite: 8].
* 📄 `test_nn.py` - Script to randomly select test images for prediction and visualize sample results [cite: 5].

---

## ⚙️ Model Architecture

The Neural Network model is built using **TensorFlow / Keras** [cite: 2], consisting of:
1. **Rescaling Layer**: Automatically normalizes pixel values from `[0, 255]` to `[0.0, 1.0]` within the model [cite: 2].
2. **Flatten Layer**: Unrolls 2D images into 1D vectors for fully-connected layers [cite: 2].
3. **Hidden Layers (MLP)**:
   * Dense Layer with 256 nodes (ReLU) + Batch Normalization + Dropout (0.4) [cite: 2].
   * Dense Layer with 128 nodes (ReLU) + Batch Normalization + Dropout (0.4) [cite: 2].
   * Dense Layer with 64 nodes (ReLU) + Dropout (0.3) [cite: 2].
4. **Output Layer**:
   * For binary classification (2 classes): Uses `sigmoid` activation with `binary_crossentropy` loss [cite: 2].
   * For multiclass classification: Uses `softmax` activation with `sparse_categorical_crossentropy` loss [cite: 2].
5. **Optimizer & Callbacks**:
   * Optimizer: Adam (Learning Rate = 1e-3) [cite: 2].
   * Early Stopping: Stops training when validation loss stops improving for 5 epochs and restores the best weights [cite: 2].
   * ReduceLROnPlateau: Halves the learning rate when validation loss plateaus for 3 epochs [cite: 2].

---

## 🚀 Getting Started

### 1. Install Required Libraries
Ensure Python and the required libraries are installed:
```bash
pip install tensorflow numpy opencv-python scikit-learn matplotlib
```

### 2. Prepare the Dataset
Place your image dataset in the following folder structure (e.g., the `AlienImages` folder located alongside the source code) [cite: 6, 8]:
```text
AlienImages/
├── alien/
│   ├── image1.jpg
│   └── ...
└── predator/
    ├── image2.jpg
    └── ...
```

### 3. Run the Main Script for Training and Evaluation
```bash
python main.py
```
The script will perform the following steps [cite: 8]:
* **Step 1:** Load images and automatically detect classes from subdirectories [cite: 6, 8].
* **Step 2:** Preprocess and convert image features [cite: 3, 8].
* **Step 3:** Split dataset into Train (70%), Validation (10%), and Test (20%) sets [cite: 4, 8].
* **Step 4:** Build and train the Neural Network model, saving it to `outputs/nn_model.keras` [cite: 2, 8].
* **Step 5 & 6:** Perform predictions, evaluate the model, and save the confusion matrix and training history plots [cite: 7, 8].

### 4. Run Random Sample Testing
After running `main.py`, you can test random images and view predictions using [cite: 5]:
```bash
python test_nn.py
```

---
*Developed by: Wit Suk [cite: 1].*
