python -c "
content = '''# 🕷️ CNN Image Classification: Spider vs Centipede 🐛

ระบบประมวลผลและจำแนกภาพถ่ายแมงมุม (Spider) และตะขาบ (Centipede) ด้วยเทคโนโลยี **Convolutional Neural Networks (CNN)** ผ่านกระบวนการ End-to-End ตั้งแต่การดึงภาพ, การทำ Preprocessing, การแบ่งข้อมูลแบบ Stratified, การเทรนโมเดลด้วย TensorFlow/Keras พร้อมด้วย Callback ป้องกัน Overfitting ไปจนถึงการประเมินผลและการทดสอบภาพจริง

---

## 📌 สถาปัตยกรรมระบบ (System Architecture)

โครงสร้างการทำงานถูกแบ่งออกเป็น 6 ขั้นตอนหลักแบบเป็นสัดส่วน (Modular Structure):

\`\`\`text
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
\`\`\`

---

## 📁 โครงสร้างโปรเจกต์ (Directory Structure)

\`\`\`text
lab7/
├── PetImages/                   # โฟลเดอร์เก็บรูปภาพจำแนกตามคลาส
│   ├── Spider/                  # รูปภาพแมงมุม
│   └── Centipede/               # รูปภาพตะขาบ
├── classification/              # โฟลเดอร์ซอร์สโค้ดหลัก
│   ├── data_loader.py           # ฟังก์ชันอ่านไฟล์ภาพจาก Disk
│   ├── preprocessing.py         # ฟังก์ชันจัดการ Color Space และ Resizing
│   ├── split_data.py            # ฟังก์ชันตัดแบ่ง Train/Val/Test Sets
│   ├── cnn_model.py             # โครงสร้างโมเดล CNN และการเทรน
│   ├── evaluate.py              # การแสดงผล Metrics, Matrix และ Curves
│   ├── main.py                  # สคริปต์หลักสำหรับรันกระบวนการทั้งหมด
│   ├── test_cnn.py              # สคริปต์สุ่มรูปภาพจาก Test Set มาสแกนทายผล
│   └── outputs/                 # โฟลเดอร์บันทึกไฟล์ Artifacts ทั้งหมดที่สร้างขึ้น
│       ├── cnn_model.keras      # ไฟล์น้ำหนักโมเดลที่เทรนสำเร็จ
│       ├── classes.json         # ชื่อคลาสของข้อมูล
│       ├── history.json         # ประวัติ Loss/Accuracy ในการเทรน
│       ├── training_history.png # กราฟเปรียบเทียบ Train vs Validation
│       ├── confusion_matrix.png # เมทริกซ์การจำแนกความถูกต้อง
│       └── prediction_sample.png# ตัวอย่างภาพทดสอบพร้อมผลการทาย
└── README.md
\`\`\`

---

## 🛠️ รายละเอียดโครงสร้างโมเดล CNN (\`cnn_model.py\`)

โมเดลออกแบบมาเพื่อป้องกันการจดจำชุดข้อมูลฝึกซ้อม (Overfitting) ในภาพขนาดเล็กโดยเฉพาะ:

1. **Input Normalization Layer:** ใช้ \`Rescaling(1./255)\` ปรับค่าพิกเซลจาก $0-255$ ให้เป็นช่วง $0-1$ ภายในตัวโมเดลโดยตรง
2. **Data Augmentation Layer:** ใส่ \`RandomFlip(\"horizontal\")\` และ \`RandomRotation(0.05)\` เพื่อสุ่มกลับด้านและหมุนภาพเล็กน้อยเฉพาะช่วงเทรน
3. **Convolutional Extractor (3 Blocks):**
   * **Block 1:** Conv2D (32 filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)
   * **Block 2:** Conv2D (64 filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)
   * **Block 3:** Conv2D (128 filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)
4. **Dense Classifier:** 
   * \`Flatten()\` + \`Dense(128, activation='relu')\`
   * \`Dropout(0.4)\` เพื่อตัดการเชื่อมต่อแบบสุ่ม ช่วยให้โมเดลกระจายการเรียนรู้
   * \`Dense(num_classes, activation='softmax')\` สำหรับ Output Probability
5. **Optimizer & Loss:** ใช้ \`Adam(learning_rate=1e-4)\` ร่วมกับ \`sparse_categorical_crossentropy\`
6. **Training Callbacks:**
   * \`EarlyStopping\`: สั่งหยุดเทรนเมื่อ \`val_loss\` ไม่ดีขึ้นติดต่อกันตามที่กำหนด
   * \`ReduceLROnPlateau\`: ลด Learning Rate ลงอัตโนมัติเมื่อค่า Loss เริ่มชะลอตัว

---

## 🚀 ขั้นตอนการติดตั้งและการใช้งาน (Getting Started)

### 1. ความต้องการของระบบ (Requirements)
* Python 3.10+
* TensorFlow 2.x
* OpenCV (\`opencv-python\`)
* Scikit-Learn
* Matplotlib
* NumPy

ติดตั้ง Library ทั้งหมดได้ผ่านคำสั่ง:
\`\`\`bash
pip install tensorflow opencv-python scikit-learn matplotlib numpy
\`\`\`

### 2. การจัดเตรียมข้อมูลภาพ
วางรูปภาพไว้ในโฟลเดอร์ \`PetImages\` โดยแยกโฟลเดอร์ตามชื่อคลาส:
\`\`\`text
PetImages/
├── Spider/
└── Centipede/
\`\`\`

### 3. การรันโมเดล (Training & Evaluation)
สั่งรันสคริปต์หลักเพียงคำสั่งเดียว ระบบจะประมวลผลตั้งแต่ต้นจนจบ:
\`\`\`bash
python classification/main.py
\`\`\`

### 4. การทดสอบสุ่มทำนายภาพ (Inference)
หลังเทรนสำเร็จ สั่งรันสคริปต์ทดสอบสุ่มภาพจาก Test Set มาตรวจผล:
\`\`\`bash
python classification/test_cnn.py
\`\`\`

---

## 📊 ผลลัพธ์และการประเมินผล (Outputs & Metrics)

เมื่อรันระบบสำเร็จ ไฟล์ผลลัพธ์จะถูกนำมาเก็บไว้ในโฟลเดอร์ \`classification/outputs/\` โดยอัตโนมัติ:

| ไฟล์ผลลัพธ์ | รายละเอียด |
| :--- | :--- |
| **\`training_history.png\`** | กราฟแสดงแนวโน้ม Accuracy และ Loss ระหว่าง Train กับ Validation เพื่อยืนยันว่าไม่เกิด Overfit |
| **\`confusion_matrix.png\`** | ตาราง Confusion Matrix แสดงจำนวนรูปภาพที่ทายถูกและทายผิดในแต่ละคลาส |
| **\`prediction_sample.png\`** | ภาพตัวอย่างที่ถูกสุ่มทดสอบ แสดงคลาสจริง (True) คลาสที่ทาย (Pred) และค่าความมั่นใจ (%) |
| **\`cnn_model.keras\`** | ไฟล์โมเดลที่บันทึกค่า Weights ที่ดีที่สุด (Best State) พร้อมนำไป Deploy ใช้งานต่อ |
'''

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('สร้างไฟล์ README.md เรียบร้อยแล้ว!')
"
