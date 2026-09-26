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
## 📁 โครงสร้างโปรเจกต์ (Directory Structure)

```
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
```
## 🛠️ รายละเอียดโครงสร้างโมเดล CNN

โมเดลออกแบบมาเพื่อป้องกันการจดจำชุดข้อมูลฝึกซ้อม (Overfitting) ในภาพขนาดเล็กโดยเฉพาะ:

```
1.Input Normalization Layer: ใช้ Rescaling(1./255) ปรับค่าพิกเซลจาก $0-255$ ให้เป็นช่วง $0-1$ ภายในตัวโมเดลโดยตรง
2.Data Augmentation Layer: ใส่ RandomFlip("horizontal") และ RandomRotation(0.05) เพื่อสุ่มกลับด้านและหมุนภาพเล็กน้อยเฉพาะช่วงเทรน
3.Convolutional Extractor (3 Blocks):Block 1: Conv2D (32 filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)Block 2: Conv2D (64 filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)Block 3: Conv2D (128 filters, $3 \times 3$, ReLU) + MaxPooling2D ($2 \times 2$)
4.Dense Classifier:Flatten() + Dense(128, activation='relu')Dropout(0.4) เพื่อตัดการเชื่อมต่อแบบสุ่ม ช่วยให้โมเดลกระจายการเรียนรู้Dense(num_classes, activation='softmax') สำหรับ Output Probability
5.Optimizer & Loss: ใช้ Adam(learning_rate=1e-4) ร่วมกับ sparse_categorical_crossentropy
6.Training Callbacks:EarlyStopping: สั่งหยุดเทรนเมื่อ val_loss ไม่ดีขึ้นติดต่อกันตามที่กำหนดReduceLROnPlateau: ลด Learning Rate ลงอัตโนมัติเมื่อค่า Loss เริ่มชะลอตัว
```
