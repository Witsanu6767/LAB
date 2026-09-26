# 🕷️ CNN Image Classification: Spider vs Centipede 🐛

ระบบประมวลผลและจำแนกภาพถ่ายแมงมุม (Spider) และตะขาบ (Centipede) ด้วยเทคโนโลยี **Convolutional Neural Networks (CNN)** ผ่านกระบวนการ End-to-End ตั้งแต่การดึงภาพ, การทำ Preprocessing, การแบ่งข้อมูลแบบ Stratified, การเทรนโมเดลด้วย TensorFlow/Keras พร้อมด้วย Callback ป้องกัน Overfitting ไปจนถึงการประเมินผลและการทดสอบภาพจริง

---

## 📌 สถาปัตยกรรมระบบ (System Architecture)

โครงสร้างการทำงานถูกแบ่งออกเป็น 6 ขั้นตอนหลักแบบเป็นสัดส่วน (Modular Structure):
```text
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
