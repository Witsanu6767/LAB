# 🕷️ CNN Image Classification: Spider vs Centipede 🐛

ระบบประมวลผลและจำแนกภาพถ่ายแมงมุม (Spider) และตะขาบ (Centipede) ด้วยเทคโนโลยี **Convolutional Neural Networks (CNN)** ผ่านกระบวนการ End-to-End ตั้งแต่การดึงภาพ, การทำ Preprocessing, การแบ่งข้อมูลแบบ Stratified, การเทรนโมเดลด้วย TensorFlow/Keras พร้อมด้วย Callback ป้องกัน Overfitting ไปจนถึงการประเมินผลและการทดสอบภาพจริง

---

## 📌 สถาปัตยกรรมระบบ (System Architecture)

โครงสร้างการทำงานถูกแบ่งออกเป็น 6 ขั้นตอนหลักแบบเป็นสัดส่วน (Modular Structure):

```
[ PetImages Data ] 
       │
       ▼
 1. data_loader.py       ──► โหลดรูปภาพ BGR, Resize เบื้องต้น
       │
       ▼
 2. preprocessing.py     ──► แปลงเป็น RGB & ปรับฟอร์แมต NumPy Array (uint8)
       │
       ▼
 3. split_data.py        ──► แบ่ง Train / Validation / Test ด้วย Stratified Sampling
       │
       ▼
 4. cnn_model.py         ──► เทรนด้วย CNN + Data Augmentation + Dropout + Callbacks
       │
       ▼
 5. evaluate.py          ──► คำนวณ Accuracy, Confusion Matrix & กราฟ Training History
       │
       ▼
 6. test_cnn.py          ──► สุ่มภาพจาก Test Set มาทดสอบทำ Prediction และแสดงผล
```

## 📁 โครงสร้างโปรเจกต์ (Project Directory)

```
lab7/
├── PetImages/                   # โฟลเดอร์ชุดข้อมูลภาพหลัก
│   ├── Spider/                  # ภาพถ่ายแมงมุม
│   └── Centipede/               # ภาพถ่ายตะขาบ
├── classification/              # โซนซอร์สโค้ดการทำงาน
│   ├── data_loader.py           # โหลดไฟล์ภาพและคัดกรองไฟล์ที่เสียหาย
│   ├── preprocessing.py         # แปลง Color Space (BGR->RGB) และ Format Data
│   ├── split_data.py            # สเกลการแบ่งข้อมูลเป็นสัดส่วนที่สมดุล
│   ├── cnn_model.py             # โครงสร้างชั้น Neural Network และระบบ Training
│   ├── evaluate.py              # คำนวณ Accuracy, Report และพล็อตภาพกราฟ
│   ├── main.py                  # สคริปต์หลักสั่งรันทั้ง Pipeline
│   ├── test_cnn.py              # สคริปต์ทดสอบสุ่มภาพจาก Test Set ทำ Prediction
│   └── outputs/                 # ผลลัพธ์และโมเดลที่ถูกบันทึกหลังรันสำเร็จ
│       ├── cnn_model.keras      # ไฟล์น้ำหนักโมเดล CNN (Best Weights)
│       ├── classes.json         # ดัชนีชื่อคลาสข้อมูล
│       ├── history.json         # ค่า Loss และ Accuracy ในแต่ละ Epoch
│       ├── training_history.png # กราฟแสดงการเรียนรู้ (Learning Curves)
│       ├── confusion_matrix.png # แผนภูมิ Confusion Matrix
│       └── prediction_sample.png# ตัวอย่างผลลัพธ์การสุ่มทดสอบทำนายภาพ
└── README.md
```

---

## 🛠️ โครงสร้างโมเดล CNN (`cnn_model.py`)

ออกแบบเพื่อเน้นประสิทธิภาพและป้องกันปัญหา **Overfitting** สำหรับชุดข้อมูลขนาดเล็ก:

* **In-Model Normalization:** ใช้ชั้น \`Rescaling(1./255)\` เพื่อปรับช่วงพิกเซล ($0–255 \rightarrow 0–1$) โดยอัตโนมัติในโมเดล
* **Data Augmentation:** เพิ่ม \`RandomFlip(\"horizontal\")\` และ \`RandomRotation(0.05)\` เพื่อเพิ่มความหลากหลายของภาพฝึกซ้อม
* **Feature Extraction (3 Blocks):**
  * **Block 1:** Conv2D (32 Filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)
  * **Block 2:** Conv2D (64 Filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)
  * **Block 3:** Conv2D (128 Filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)
* **Classification Head:** 
  * \Flatten()\ + \Dense(128, activation='relu')\
  * \Dropout(0.4)\ ตัดการเชื่อมต่อสุ่มป้องกันการจดจำข้อผิดพลาด
  * \Dense(num_classes, activation='softmax')\ ประมวลผลความน่าจะเป็นของคลาส
* **Optimization & Callbacks:**
  * Optimizer: **Adam** (Learning Rate = \1e-4\)
  * Loss Function: **Sparse Categorical Crossentropy**
  * \EarlyStopping\: หยุดการฝึกเมื่อ Validation Loss ไม่ดีขึ้นต่อเนื่อง
  * \ReduceLROnPlateau\: ปรับลด Learning Rate อัตโนมัติเมื่อค่า Loss เริ่มคงที่

---

## 🚀 ขั้นตอนการติดตั้งและการใช้งาน (Quick Start)

### 1. ติดตั้ง Dependencies
```bash
pip install tensorflow opencv-python scikit-learn matplotlib numpy
```

### 2. การจัดวางไฟล์ชุดข้อมูล
นำโฟลเดอร์ภาพวางไว้ในไดเรกทอรี \PetImages/\ แยกตามชื่อคลาส:
```
PetImages/
├── Spider/
└── Centipede/
```

### 3. รันกระบวนการเรียนรู้และประเมินผล (Execution)
สั่งรัน pipeline หลักเพียงคำสั่งเดียว ระบบจะเริ่มทำงานตั้งแต่ขั้นตอนที่ 1 ถึง 6:
```bash
python classification/main.py
```

### 4. ทดสอบสุ่มทำนายภาพ (Inference Test)
ทดสอบนำโมเดลที่เทรนเสร็จแล้วมารันทำนายภาพสุ่มจากชุด Test Set:
```bash
python classification/test_cnn.py
```

---

## 📊 รายงานผลลัพธ์ (Artifacts & Output Files)

หลังรันสคริปต์เรียบร้อย ผลลัพธ์ทั้งหมดจะถูกสร้างไว้ที่ \`classification/outputs/\`:

| ไฟล์ผลลัพธ์ | ประเภท | คำอธิบาย |
| :--- | :--- | :--- |
| **\training_history.png\** | Image | กราฟแนวโน้ม Accuracy & Loss เปรียบเทียบระหว่าง Training และ Validation Sets |
| **\confusion_matrix.png\** | Image | แสดงสถิติจำนวนภาพที่ทายถูกต้องและผิดพลาดของแต่ละคลาส |
| **\prediction_sample.png\** | Image | ตัวอย่างผลการสุ่มทำนาย พร้อมแสดงระดับความมั่นใจ (%) และสถานะ Correct/Wrong |
| **\cnn_model.keras\** | Model File | โมเดลสำเร็จรูปที่พร้อมนำไปปรับใช้ (Deploy) ร่วมกับระบบอื่น |
'''
