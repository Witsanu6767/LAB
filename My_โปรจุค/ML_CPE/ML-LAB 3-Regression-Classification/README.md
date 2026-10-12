# 🏴‍☠️ One Piece Bounty Prediction Model

This project utilizes character data from the anime **One Piece** to build a **Linear Regression** Machine Learning model to predict character **Bounties** based on various features such as Age, Height, and Devil Fruit consumption.

---

## 🎯 Goal
To evaluate how accurately physical traits and special abilities can predict character bounties in the One Piece universe by comparing three model variations.

---

## 🛠️ Models Built

1. **Simple Linear Regression (Single Feature)**
   * Predicts bounty using **Age** alone.
   * *Result:* Age alone proved **inaccurate** for predictions, as several younger characters (e.g., Luffy) hold significantly higher bounties compared to older characters.

2. **Multiple Linear Regression (Multi-Feature)**
   * Incorporates additional features: **Height_cm** and **Devil Fruit Type** (encoded into numerical values using One-Hot Encoding).
   * *Result:* Enabled the model to capture complexity, significantly improving performance compared to using age alone.

3. **Bounty Prediction System (Custom Character Predictor)**
   * A simplified pipeline taking **Age** and **Height** inputs to instantly estimate bounties for newly created characters.

---

## 📊 Usage Example

You can modify `input_age` and `input_height` to predict the bounty of a new character:

```python
# Input custom character features for prediction
input_age = 30
input_height = 314

new_character = pd.DataFrame({'Age': [input_age], 'Height_cm': [input_height]})
predicted_bounty = bounty_system.predict(new_character)

print(f"🔮 Predicted Bounty: {predicted_bounty[0]:,.2f} Berries")
```

---

# 🍎 One Piece Devil Fruit Classifier (Logistic Regression)

This project uses **One Piece** character data to construct a **Logistic Regression** Classification model predicting whether a character **"Has a Devil Fruit" (Binary Classification)**, using physical traits such as **Height_cm** and **Age**.

---

## 🎯 Goal
Investigate the relationship between height, age, and the probability of being a Devil Fruit user, while visualizing the model's **Decision Boundary**.

---

## 🛠️ Workflow

1. **Data Preparation & Labeling**
   * Transformed the `Devil_Fruit` column into a binary target:
     * `1` = Has Devil Fruit
     * `0` = No Devil Fruit
2. **Model Training**
   * Trained a **Logistic Regression** model using `class_weight='balanced'` to address class imbalance.
3. **Model Evaluation**
   * Assessed performance using **Confusion Matrix**, **Accuracy**, and **Classification Report (Precision, Recall, F1-Score)**.
4. **Decision Boundary Visualization**
   * Plotted regions and boundary lines separating Devil Fruit users from non-users.
5. **Interactive Prediction System**
   * Built an interactive system predicting the likelihood of Devil Fruit ownership along with probability percentages for new characters.

---

## 📊 Usage Example

Input custom **Height** and **Age** values to predict the likelihood of a character having a Devil Fruit:

```python
# Input height and age to predict
input_height = 180
input_age = 25

new_character = pd.DataFrame({'Height_cm': [input_height], 'Age': [input_age]})

# Predict class and probabilities
predicted_class = clf_model.predict(new_character)[0]
predicted_proba = clf_model.predict_proba(new_character)[0]

status = "Has Devil Fruit 🍎" if predicted_class == 1 else "No Devil Fruit ⚔️"
print(f"Prediction Result: {status}")
print(f"Devil Fruit Probability: {predicted_proba[1]*100:.2f}%")
```

---

# ⚔️ One Piece ML Pipeline: Regression vs Classification Benchmark

An end-to-end Machine Learning benchmark comparing continuous numerical prediction (**Regression** for Bounty estimation) against categorical group classification (**Classification** for Devil Fruit user detection).

---

## 🎯 Project Overview

This benchmark evaluates performance across three core areas:
1. **Simple vs Multiple Linear Regression**: Comparing single-feature (Age) vs multi-feature (Age, Height, Devil Fruit Type) bounty predictions.
2. **Train vs Test Evaluation (Overfitting Check)**: Assessing model stability for overfitting or underfitting.
3. **Regression vs Classification Task Comparison**: Summarizing theoretical and evaluation differences between the two ML approaches.

---

## 🛠️ Tested Models & Features

* **Simple Linear Regression**
  * **Target:** `Bounty` (Continuous)
  * **Features:** `Age`
* **Multiple Linear Regression**
  * **Target:** `Bounty` (Continuous)
  * **Features:** `Age`, `Height_cm`, `Devil_Fruit_Type` (One-Hot Encoded)
* **Logistic Regression Classifier**
  * **Target:** `Has_Devil_Fruit` (Discrete: 0 or 1)
  * **Features:** `Age`, `Height_cm` (`class_weight='balanced'`)

---

## 📊 Benchmark Results

### 1. Regression Comparison (Simple vs Multiple)

| Metric | Simple Linear Regression | Multiple Linear Regression |
| :--- | :---: | :---: |
| **MAE (Berries)** | High average error | Significantly reduced error |
| **RMSE (Berries)** | High error driven by extreme outliers | Improved robustness against outliers |
| **R² Score (Test)** | Negative / Poor fit | Explains a broader scope of variance |

> **Conclusion:** Incorporating height and Devil Fruit type improved the predictive performance of the Multiple Linear Regression model significantly.

---

### 2. Task Overview (Regression vs Classification)

| Aspect | Regression (Bounty Prediction) | Classification (Devil Fruit Detection) |
| :--- | :--- | :--- |
| **Target Type** | Continuous numerical values | Discrete categories/classes |
| **Output** | Currency values (Berries) | Status (0 = No Power, 1 = Has Power) |
| **Primary Metrics** | MAE, RMSE, $R^2$ Score | Accuracy, Precision, Recall, F1-Score |

---

## 📦 Tech Stack

* **Python 3.x**
* **Data Processing:** `Pandas`, `NumPy`
* **Machine Learning:** `scikit-learn` (`LinearRegression`, `LogisticRegression`, `train_test_split`, `metrics`)
* **Data Visualization:** `Matplotlib`

---

💡 *Note: This project demonstrates that when working with anime data characterized by extreme volatility and outliers, selecting relevant features and matching model architecture to the task type are critical to model performance.*
