# 🐾 VGG Image Recognition Project (PyTorch & OpenCV)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

โปรเจกต์นี้เป็นการพัฒนาโมเดล Deep Learning ด้วยโครงสร้างแบบ **VGG** (สถาปัตยกรรมสไตล์ VGG-16) ผ่านไลบรารี **PyTorch** ร่วมกับ **OpenCV** สำหรับงานจำแนกภาพ (Image Classification) เช่น การจำแนกภาพสุนัขและแมว (Cat vs Dog) อย่างมีประสิทธิภาพ

---

## 📂 โครงสร้างและหน้าที่ของไฟล์ในโปรเจกต์

ระบบถูกออกแบบในรูปแบบโมดูลาร์ (Modular Design) เพื่อให้ง่ายต่อการดูแลรักษาและต่อยอด:

| ไฟล์หลัก | หน้าที่หลัก |
| :--- | :--- |
| **`main.py`** | จุดเริ่มต้นหลัก (Entry Point) ควบคุม Pipeline ทั้งหมดตั้งแต่อ่านข้อมูลจนถึงประเมินผล |
| **`data_loader.py`** | โหลดภาพจากโครงสร้างโฟลเดอร์อัตโนมัติ (Automatic Class Detection) พร้อมกรองภาพเสีย |
| **`preprocessing.py`** | แปลงสีภาพจาก BGR เป็น RGB และปรับขนาดภาพล่วงหน้าด้วย `cv2.INTER_AREA` |
| **`split_data.py`** | แบ่งชุดข้อมูลออกเป็น Train, Validation และ Test แบบรักษาสัดส่วนคลาส (Stratified Split) |
| **`vgg_model.py`** | สร้างสถาปัตยกรรม `PyTorchVGG`, จัดการการเทรน/ทำนายผล และรองรับการทำงานบน GPU/CPU |
| **`evaluate.py`** | ประเมินผลโมเดล (Accuracy, Classification Report, Confusion Matrix) และสร้างกราฟสถิติ |
| **`test_vgg.py`** | สคริปต์สุ่มทดสอบภาพจากชุด Test พร้อมแสดงผลลัพธ์ความมั่นใจ (Confidence) ออกมาเป็นภาพกราฟิก |

---

## ⚙️ ขั้นตอนการทำงาน (Pipeline Flow)

1. **📥 Loading Data:** อ่านข้อมูลภาพจากไดเรกทอรีและจำกัดขนาดความละเอียดตามที่กำหนด
2. **🎨 Preprocessing:** จัดรูปแบบฟีเจอร์และปรับแต่งช่วงข้อมูลให้พร้อมเข้าสู่กระบวนการเรียนรู้
3. **✂️ Splitting:** แบ่งสัดส่วนข้อมูลสำหรับฝึกสอน ตรวจสอบ และทดสอบระบบ
4. **🚀 Training:** ฝึกสอนโมเดล VGG บน PyTorch พร้อมบันทึกค่าน้ำหนักที่ดีที่สุดลงในไฟล์ `outputs/vgg_model.pt`
5. **📊 Evaluation & Testing:** ประเมินความแม่นยำ พร้อมสร้างกราฟ Confusion Matrix และ Training History เก็บไว้ที่โฟลเดอร์ `outputs/`

---

## 🚀 วิธีการใช้งาน

### 1. การเตรียมข้อมูล
สร้างโฟลเดอร์เก็บข้อมูลภาพ (ตัวอย่างเช่น โฟลเดอร์ `PetImages` อยู่ในระดับเดียวกับโปรเจกต์) โดยภายในมีโฟลเดอร์ย่อยตามชื่อคลาส:
```text
PetImages/
├── Cat/
│   ├── 001.jpg
│   └── 002.jpg
└── Dog/
    ├── 001.jpg
    └── 002.jpg
```

### 2. การฝึกสอนโมเดล (Training)
ตรวจสอบและปรับแต่งค่าพารามิเตอร์ต่างๆ เช่น `IMG_SIZE`, `EPOCHS`, หรือ `BATCH_SIZE` ในไฟล์ `main.py` จากนั้นรันคำสั่ง:
```bash
python main.py
```

### 3. การทดสอบโมเดล (Inference)
หลังจากเทรนโมเดลเรียบร้อยแล้ว สามารถสุ่มภาพในชุดทดสอบมาทำนายผลและสร้างภาพตัวอย่างได้ด้วยคำสั่ง:
```bash
python test_vgg.py
```

---
*💡 ผลลัพธ์และกราฟสถิติต่างๆ จะถูกบันทึกอัตโนมัติไว้ภายในโฟลเดอร์ `outputs/` ทันทีหลังรันโปรแกรม*
