# ML-05-SVM

โปรเจกต์นี้เป็นการทดลองใช้ **Support Vector Machine (SVM)** สำหรับจำแนกรูปตัวเลขที่เขียนด้วยลายมือ

## Dataset

ใช้ **Handwritten Digits Dataset** จาก `scikit-learn`

Dataset ประกอบด้วยรูปตัวเลขตั้งแต่ `0-9` จำนวน 1,797 รูป โดยแต่ละรูปมีขนาด 8x8 pixels

ในโค้ดใช้

```python
from sklearn.datasets import load_digits
```

เพื่อโหลด Dataset โดยไม่ต้องดาวน์โหลดไฟล์เพิ่ม

## หลักการทำงาน

```text
Handwritten Digits Dataset
          ↓
    Preprocess Data
          ↓
      Split Data
          ↓
       Train SVM
          ↓
      Prediction
          ↓
      Evaluation
```

SVM จะเรียนรู้รูปแบบของตัวเลขจาก Training Data แล้วนำไปทำนายข้อมูลใหม่ว่าเป็นตัวเลข `0-9` ตัวไหน

## ขั้นตอนการทำงาน

### 1. Load Data
โหลด Handwritten Digits Dataset จาก `scikit-learn`

### 2. Preprocess
ปรับข้อมูลให้อยู่ในรูปแบบที่เหมาะสมสำหรับนำไป Train

### 3. Split Data
แบ่งข้อมูลออกเป็น Training Data และ Test Data

### 4. Train SVM
นำ Training Data ไปสร้างโมเดล SVM

### 5. Prediction
ใช้โมเดลที่ Train แล้วทำนายตัวเลขจาก Test Data

### 6. Evaluation
ตรวจสอบผลด้วย Accuracy และ Confusion Matrix

## Output

ผลลัพธ์จะถูกเก็บไว้ในโฟลเดอร์

```text
output/
```

ตัวอย่างไฟล์

```text
confusion_matrix.png
prediction_sample.png
svm_model.pkl
scaler.pkl
```

## วิธี Run

ติดตั้ง Library ก่อน

```bash
pip install -r requirements.txt
```

จากนั้น Run

```bash
python main.py
```

สำหรับทดสอบตัวอย่าง Prediction

```bash
python test_svm.py
```

## ผลการทดลอง

จากการทดลอง SVM สามารถจำแนกตัวเลขใน Test Data ได้ Accuracy ประมาณ **97.78%**

## สรุป

โปรเจกต์นี้เป็นการทดลองใช้ SVM ในการจำแนกรูปตัวเลขที่เขียนด้วยลายมือ โดยโมเดลจะเรียนรู้จากข้อมูล Training แล้วนำไปทำนายข้อมูลใหม่ว่าเป็นตัวเลขอะไร
