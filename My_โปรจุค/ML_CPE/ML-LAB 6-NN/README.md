# Lab 6: Neural Network and Applications (Alien vs Predator Image Classification)

This repository contains the source code and documentation for **Lab 6: Neural Network and Applications**, focusing on building a Fully-Connected Neural Network (MLP) using Python and TensorFlow/Keras to classify image datasets.

---

## 1. Objectives
* Understand the core principles of Neural Networks (NN) and their practical applications in classification and prediction.
* Develop a complete workflow—including data loading, preprocessing, model training, and performance evaluation—to build a functional prototype utilizing a Neural Network.

---

## 2. Code Structure & Modules

The project is organized into modular Python scripts for clarity and maintainability:

* **`data_loader.py`**: Automatically detects classes from subdirectories, loads images, and resizes them to standard dimensions while handling corrupt or unreadable files.
* **`preprocessing.py`**: Casts images into efficient `uint8` arrays to save memory and utilizes an internal Keras `Rescaling` layer to scale pixel values from `[0, 255]` to `[0, 1]`.
* **`split_data.py`**: Splits the dataset into Training, Validation, and Testing sets using stratified sampling (`train_test_split`).
* **`nn_model.py`**: Builds a Fully-Connected Neural Network (MLP) architecture equipped with regularization techniques such as Batch Normalization, Dropout, Early Stopping, and Learning Rate Reduction (`ReduceLROnPlateau`).
* **`evaluate.py`**: Computes accuracy scores, generates a classification report, prints the confusion matrix, and plots training history curves (Loss/Accuracy) as well as the confusion matrix image.
* **`test_nn.py`**: Randomly samples images from the test set to evaluate predictions visually in a grid format, displaying confidence scores.
* **`main.py`**: The main entry point that orchestrates the entire pipeline from data loading, preprocessing, dataset splitting, model training, and evaluation.

---

## 3. Experimental Outputs

Running the pipeline generates and saves the following artifacts inside the `outputs/` directory:

* **Accuracy Scores**: Quantitative performance metrics evaluated on the test dataset.
* **Confusion Matrix & Classification Report**: Detailed class-wise metrics saved as `outputs/confusion_matrix.png`.
* <img width="750" height="750" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d7e98a59-34ee-4cf1-9b8f-f178b9b07ba1" />
* **Training History**: Accuracy and loss curves across epochs saved as `outputs/training_history.png`.
* <img width="1650" height="600" alt="training_history" src="https://github.com/user-attachments/assets/165de932-724a-4cf2-b4be-41f3d5bf8635" />
* **Prediction Samples**: A 2x2 grid displaying random test samples with true labels, predicted labels, and confidence levels, saved as `outputs/prediction_sample.png`.
* <img width="1020" height="1200" alt="prediction_sample" src="https://github.com/user-attachments/assets/b98a8b95-d2d9-40e3-839c-2a7a0c6746f3" />

---

## 4. Conclusion
Through this lab, we successfully applied a Neural Network to an image classification task. The implementation demonstrates how proper hyperparameter configuration, validation splits, and regularization techniques (like Dropout and Early Stopping) effectively prevent overfitting and yield robust predictive performance for real-world applications.
