# ML-07-CNN 

## รายละเอียด Lab

Lab นี้เป็นการทดลองใช้ Convolutional Neural Network (CNN) สำหรับจำแนกรูปภาพ โดยเปลี่ยน Dataset จากตัวอย่างเดิมเป็นการจำแนก **Rice** และ **Corn**

จุดประสงค์คือให้โมเดลเรียนรู้จากรูปภาพ แล้วทำนายรูปที่ไม่เคยเห็นมาก่อนว่าเป็น Rice หรือ Corn

## Dataset

มี 2 Class

- Rice — เมล็ดข้าว
- Corn — ข้าวโพด

โครงสร้าง Dataset:

```text
Dataset/
├── Corn/
└── Rice/
```

## โครงสร้างโปรเจกต์

```text
m-project/
├── main.py
├── data_loader.py
├── preprocessing.py
├── split_data.py
├── cnn_model.py
├── evaluate.py
├── test_cnn.py
├── run.bat
├── requirements.txt
├── README.md
├── Dataset/
│   ├── Corn/
│   └── Rice/
└── outputs/
```

โครงสร้างหลักยังยึดตามโปรเจกต์ Lab 07 เดิมของอาจารย์ โดยเปลี่ยน Dataset และแก้ส่วนอ่านรูปให้รองรับ path ภาษาไทยบน Windows

## วิธีติดตั้ง

แนะนำให้ใช้ **Python 3.12** สำหรับ TensorFlow

เปิด Terminal ในโฟลเดอร์ `m-project` แล้วใช้:

```bash
python -m pip install -r requirements.txt
```

## วิธี Run

เปิด `main.py` แล้วกด **Run Python File** หรือใช้ Terminal:

```bash
python main.py
```

สามารถดับเบิลคลิก `run.bat` เพื่อ Run ได้เช่นกัน

## ขั้นตอนการทำงาน

1. โหลดรูปจาก Dataset
2. Resize รูปเป็น 100 x 100
3. แบ่งข้อมูลเป็น Training, Validation และ Testing
4. สร้างและ Train โมเดล CNN
5. ทดสอบโมเดล
6. ประเมินผลด้วย Accuracy, Classification Report และ Confusion Matrix

## Output

หลัง Run เสร็จ ผลลัพธ์จะอยู่ในโฟลเดอร์ `outputs` เช่น

- `cnn_model.keras` — โมเดลที่ Train แล้ว
- `history.json` — ข้อมูลการ Train
- `confusion_matrix.png` — Confusion Matrix
- `training_history.png` — กราฟ Accuracy และ Loss
- `prediction_sample.png` — ผลการทำนายตัวอย่าง เมื่อ Run `test_cnn.py`
- ไฟล์ `.npy` — ข้อมูลที่ใช้ในแต่ละขั้นตอน

## หมายเหตุ

ถ้า VS Code ใช้ Python 3.14 แล้วติดตั้ง TensorFlow ไม่ได้ ให้เปลี่ยน Python Interpreter เป็น **Python 3.12 (64-bit)** แล้วติดตั้ง `requirements.txt` ใหม่

โปรเจกต์นี้ใช้รูปภาพ 2 Class คือ Rice และ Corn เพื่อให้รูปแบบการทำ Lab ยังคงเป็นการเปรียบเทียบ 2 ประเภทตามโครงสร้างเดิม
