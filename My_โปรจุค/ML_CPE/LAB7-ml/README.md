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
