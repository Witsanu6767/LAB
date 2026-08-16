# 🚗 Vehicle Data Analysis & Machine Learning Suite

This repository contains two end-to-end Machine Learning projects implemented from scratch using **NumPy**, focusing on vehicle dataset analysis:
1. **Supervised Learning:** Car Type Classification using K-Nearest Neighbors (KNN).
2. **Unsupervised Learning:** Vehicle Data Clustering using K-Means and KNN Assignment.

---

## 📌 Project 1: Supervised Car Type Classification (KNN Model)

This module implements a **K-Nearest Neighbors (KNN)** classification model built from scratch using **NumPy** to predict vehicle categories (`Type_car`) based on physical specifications and origin regions.

### 🎯 Objectives
* Implement a custom **NumPy-based KNN classifier** using array broadcasting and matrix manipulation.
* Search for the optimal $k$-neighbor hyperparameter using validation split evaluation.
* Benchmark classification accuracy against `scikit-learn` and baseline guessing.

### 🛠️ Data Pipeline & Preprocessing (`data_loader.py`)
* **Feature Selection:** Numerical attributes (`Age (Years)`, `Weight (kg)`, `Height (cm)`, `Top_Speed (kmh)`) and categorical mapping for `Primary_Region`.
* **Class Filtering:** Filters target classes with fewer than 3 samples to maintain evaluation validity.
* **Stratified Splitting:** Partitioned into **Train (60%) / Validation (20%) / Test (20%)** sets.
* **Normalization:** Applied `StandardScaler` fitted on the training set.

### 🧠 Core Algorithm (`knn_tf.py`)
1. **Euclidean Distance:** Calculates distances between feature vectors:
   $$\text{distance} = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$
2. **Top-K Selection:** Uses `np.argsort` to fetch nearest sample indices.
3. **Majority Voting:** Sums one-hot encoded neighbor labels (`np.eye`) to extract the final predicted class via `np.argmax`.

### 📊 Artifacts & Outputs (`evaluate.py`)
* `outputs/01_k_curve.png`: Validation accuracy plot across $k \in [1, 2, 3, 5, 7, 9, 11, 15]$.
* `outputs/02_confusion_matrix.png`: Multi-class confusion matrix visualization.
* `outputs/predictions.csv`: Detailed test set predictions with correctness indicators.

---

## 📌 Project 2: Unsupervised Vehicle Clustering & KNN Assigner

This module implements a custom **K-Means Clustering** algorithm built from scratch using **NumPy** to segment vehicle records, alongside a **KNN Assigner** to categorize new incoming data into established clusters.

### 🎯 Objectives
* Implement a robust **NumPy-based K-Means algorithm** featuring empty cluster protection and convergence criteria.
* Determine optimal cluster counts ($k$) using the **Elbow Method** and validate quality using the **Silhouette Score**.
* Assign unseen data to cluster centroids using a fast KNN-based assigner without retraining K-Means.

### 🛠️ Data Pipeline & Normalization (`data_loader.py`)
* **Features:** Continuous variables (`Age (Years)`, `Weight (kg)`, `Height (cm)`, `Price_THB`, `Top_Speed (kmh)`).
* **Feature Scaling:** Scaled via `StandardScaler` ($\mu = 0, \sigma = 1$) to standardize feature weights.

### 🧠 Core Algorithm (`kmeans_tf.py` & `knn_tools.py`)
* **K-Means Clustering:** Iteratively updates centroids based on cluster member means until centroid movement drops below $1 \times 10^{-4}$ or reaches max iterations.
* **KNN Cluster Assigner:** Computes distance matrices for new samples against known cluster data, returning majority cluster votes using `np.argpartition` and one-hot summation.

### 📊 Artifacts & Outputs (`visualize.py`)
* `outputs/01_elbow.png`: Inertia reduction curve across candidate cluster counts $k \in [2..8]$.
* `outputs/02_clusters.png`: 2D scatter plot visualizing vehicle clusters on `Weight` vs. `Height` axes.
* `outputs/cluster_summary.csv`: Aggregated profile metrics (mean attributes) per cluster.
* `outputs/clustered_car.csv`: Exported vehicle dataset appended with predicted `cluster` IDs.

---

## 🚀 Execution Guide

Run either pipeline via their respective entry scripts:

```bash
# Run Supervised KNN Classification
python main_classification.py

# Run Unsupervised K-Means Clustering & KNN Assignment
python main_clustering.py
```

---

## 📦 Tech Stack

* **Python 3.x**
* **Core Numerical Computing:** `NumPy`
* **Data Processing:** `Pandas`
* **Machine Learning & Metrics:** `scikit-learn` (`StandardScaler`, `train_test_split`, `metrics`, `KNeighborsClassifier`)
* **Visualization:** `Matplotlib`
