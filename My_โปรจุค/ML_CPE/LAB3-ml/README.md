# แยกLabย่อยออกจากกันเดี๋ยวเยอะเกิน
# 🏴‍☠️ One Piece Bounty Prediction Model

โปรเจกต์นี้เป็นการนำข้อมูลตัวละครจากการ์ตูน **One Piece** มาสร้างโมเดล Machine Learning ด้วยวิธี **Linear Regression** เพื่อทำนาย **ค่าหัว (Bounty)** ของตัวละครจากปัจจัยต่างๆ เช่น อายุ (Age), ส่วนสูง (Height), และการกินผลปีศาจ (Devil Fruit)

---

## 🎯 วัตถุประสงค์ (Goal)
เพื่อศึกษาว่าปัจจัยทางกายภาพและพลังพิเศษของตัวละคร สามารถนำมาใช้ทำนายค่าหัวในโลกของ One Piece ได้แม่นยำแค่ไหน โดยเปรียบเทียบการทำงานของโมเดล 3 รูปแบบ

---

## 🛠️ โมเดลที่ใช้สร้าง (Models Built)

1. **Simple Linear Regression (ใช้ 1 ปัจจัย)**
   * ใช้ **อายุ (Age)** เพียงอย่างเดียวในการทำนายค่าหัว
   * *ผลลัพธ์:* พบว่าอายุอย่างเดียว **ทำนายได้ไม่แม่นยำ** เนื่องจากตัวละครอายุน้อยบางคน (เช่น ลูฟี่) มีค่าหัวสูงมากเมื่อเทียบกับตัวละครที่อายุเยอะกว่า

2. **Multiple Linear Regression (ใช้หลายปัจจัย)**
   * เพิ่มปัจจัย **ส่วนสูง (Height_cm)** และ **ประเภทผลปีศาจ (Devil Fruit Type)** เข้ามาช่วยคำนวณ (ใช้เทคนิค One-Hot Encoding เปลี่ยนข้อมูลข้อความให้เป็นตัวเลข)
   * *ผลลัพธ์:* ช่วยให้โมเดลเข้าใจความซับซ้อนและทำนายได้ดีขึ้นกว่าการใช้อายุเพียงอย่างเดียว

3. **Bounty Prediction System (ระบบทำนายค่าหัวตัวละครใหม่)**
   * โมเดลฉบับย่อที่รับค่า **อายุ** และ **ส่วนสูง** เพื่อประมวลผลทำนายค่าหัวของตัวละครที่เราสร้างขึ้นมาใหม่ได้ทันที!

---

## 📊 ตัวอย่างโค้ดระบบทำนาย (Usage Example)

คุณสามารถลองเปลี่ยนค่า `input_age` และ `input_height` เพื่อทำนายค่าหัวของตัวละครใหม่ได้:

```python
# ใส่ข้อมูลตัวละครที่ต้องการทำนาย
input_age = 30
input_height = 314

new_character = pd.DataFrame({'Age': [input_age], 'Height_cm': [input_height]})
predicted_bounty = bounty_system.predict(new_character)

print(f"🔮 ค่าหัวที่คาดการณ์: {predicted_bounty[0]:,.2f} เบรี")
```

---

# 🍎 One Piece Devil Fruit Classifier (Logistic Regression)

โปรเจกต์นี้เป็นการนำข้อมูลตัวละคร **One Piece** มาสร้างโมเดล จำแนกประเภท (Classification) ด้วยวิธี **Logistic Regression** เพื่อทำนายว่าตัวละครนั้น **"กินผลปีศาจหรือไม่" (Has Devil Fruit)** โดยวิเคราะห์จากปัจจัยทางกายภาพ ได้แก่ **ส่วนสูง (Height_cm)** และ **อายุ (Age)**

---

## 🎯 วัตถุประสงค์ (Goal)
ศึกษาความสัมพันธ์ระหว่างส่วนสูงและอายุของตัวละครว่าส่งผลต่อโอกาสในการเป็นผู้ใช้พลังผลปีศาจหรือไม่ พร้อมทั้งจำลองเส้นแบ่งเขตการตัดสินใจ (**Decision Boundary**) ของโมเดล

---

## 🛠️ ขั้นตอนการทำงาน (Workflow)

1. **Data Preparation & Labeling**
   * เปลี่ยนแปลงข้อมูลคอลัมน์ `Devil_Fruit` ให้เป็นรูปแบบ Binary Target:
     * `1` = กินผลปีศาจ (Has Devil Fruit)
     * `0` = ไม่ได้กินผลปีศาจ (No Devil Fruit)
2. **Model Training**
   * เลือกใช้ **Logistic Regression** พร้อมกำหนด `class_weight='balanced'` เพื่อช่วยแก้ปัญหาข้อมูลสองฝั่งที่มีจำนวนไม่เท่ากัน (Class Imbalance)
3. **Model Evaluation**
   * วัดผลประสิทธิภาพด้วย **Confusion Matrix**, **Accuracy**, และ **Classification Report (Precision, Recall, F1-Score)**
4. **Decision Boundary Visualization**
   * สร้างกราฟแสดงพื้นที่และเส้นแบ่งเขตการจำแนกประเภทระหว่างกลุ่มคนกินผลปีศาจและไม่กินผลปีศาจ
5. **Interactive Prediction System**
   * ระบบทำนายโอกาสการกินผลปีศาจพร้อมระบุเปอร์เซ็นต์ความน่าจะเป็น (Probability) สำหรับตัวละครใหม่

---

## 📊 การนำระบบไปใช้งานจริง (Usage Example)

ทดลองป้อนค่า **ส่วนสูง** และ **อายุ** เพื่อทำนายโอกาสการกินผลปีศาจของตัวละครใหม่:

```python
# ใส่ข้อมูลส่วนสูงและอายุที่ต้องการทำนาย
input_height = 180
input_age = 25

new_character = pd.DataFrame({'Height_cm': [input_height], 'Age': [input_age]})

# ทำนายคลาสและความน่าจะเป็น
predicted_class = clf_model.predict(new_character)[0]
predicted_proba = clf_model.predict_proba(new_character)[0]

status = "Has Devil Fruit 🍎" if predicted_class == 1 else "No Devil Fruit ⚔️"
print(f"ผลการทำนาย: {status}")
print(f"โอกาสกินผลปีศาจ: {predicted_proba[1]*100:.2f}%")
```

---

# ⚔️ One Piece ML Pipeline: Regression vs Classification Benchmark

โปรเจกต์เปรียบเทียบประสิทธิภาพโมเดล Machine Learning แบบครบวงจร โดยนำข้อมูลตัวละคร **One Piece** มาวิเคราะห์ทั้งโจทย์การทำนายเชิงตัวเลขต่อเนื่อง (**Regression**) เพื่อประเมินค่าหัว (Bounty) และโจทย์การจำแนกกลุ่ม (**Classification**) เพื่อจำแนกผู้ใช้งานพลังผลปีศาจ (Devil Fruit User)

---

## 🎯 ภาพรวมการเปรียบเทียบ (Project Overview)

โครงการนี้มุ่งเน้นการประเมินประสิทธิภาพ 3 ประเด็นหลัก:
1. **Simple vs Multiple Linear Regression**: เปรียบเทียบผลการทำนายค่าหัวระหว่างการใช้ปัจจัยเดียว (Age) กับหลายปัจจัย (Age, Height, Devil Fruit Type)
2. **Train vs Test Evaluation (Overfitting Check)**: ตรวจสอบความเสถียรของโมเดลเพื่อประเมินภาวะ Overfitting/Underfitting
3. **Regression vs Classification Task Comparison**: สรุปข้อแตกต่างเชิงทฤษฎีและการวัดผลระหว่างสองรูปแบบ Machine Learning

---

## 🛠️ โครงสร้างโมเดลที่ใช้ทดสอบ (Models & Features)

* **Simple Linear Regression**
  * **Target:** `Bounty` (ค่าหัว - Continuous)
  * **Features:** `Age`
* **Multiple Linear Regression**
  * **Target:** `Bounty` (ค่าหัว - Continuous)
  * **Features:** `Age`, `Height_cm`, `Devil_Fruit_Type` (One-Hot Encoded)
* **Logistic Regression Classifier**
  * **Target:** `Has_Devil_Fruit` (0 หรือ 1 - Discrete)
  * **Features:** `Age`, `Height_cm` (ใช้ `class_weight='balanced'`)

---

## 📊 สรุปผลการเปรียบเทียบโมเดล (Benchmark Results)

### 1. Regression Comparison (Simple vs Multiple)

| Metric | Simple Linear Regression | Multiple Linear Regression |
| :--- | :---: | :---: |
| **MAE (Berries)** | ประเมินความคลาดเคลื่อนเฉลี่ยสูง | ความคลาดเคลื่อนลดลงอย่างมีนัยสำคัญ |
| **RMSE (Berries)** | มีค่า Error สูงจาก Outliers | ทนทานต่อ Outliers ได้ดีขึ้น |
| **R² Score (Test)** | ค่าติดลบ / ไม่สมบูรณ์ | แสดงความสัมพันธ์ของข้อมูลได้ครอบคลุมกว่า |

> **ข้อสรุป:** การเพิ่มตัวแปรส่วนสูงและประเภทผลปีศาจ ช่วยเพิ่มประสิทธิภาพในการทำนายค่าหัวของโมเดล Multiple Linear Regression ได้อย่างชัดเจน

---

### 2. Task Overview (Regression vs Classification)

| Aspect | Regression (ทำนายค่าหัว) | Classification (ตรวจจับผลปีศาจ) |
| :--- | :--- | :--- |
| **Target Type** | ตัวเลขต่อเนื่อง (Continuous) | กลุ่ม/คลาส (Discrete Categories) |
| **Output** | มูลค่าเงินเบรี (Numerical Values) | สถานะ (0 = ไม่มีพลัง, 1 = มีพลัง) |
| **Primary Metrics** | MAE, RMSE, $R^2$ Score | Accuracy, Precision, Recall, F1-Score |

---

## 📦 ไลบรารีที่ใช้งาน (Tech Stack)

* **Python 3.x**
* **Data Processing:** `Pandas`, `NumPy`
* **Machine Learning:** `scikit-learn` (`LinearRegression`, `LogisticRegression`, `train_test_split`, `metrics`)
* **Data Visualization:** `Matplotlib`

---

💡 *Note: โปรเจกต์นี้แสดงให้เห็นว่าข้อมูลประเภทอนิเมะที่มีความผันผวนสูง (Extreme Outliers) การเลือก Feature ที่เหมาะสมและการเลือกประเภทโมเดลให้ตรงกับโจทย์ มีผลอย่างมากต่อความแม่นยำของระบบ*
