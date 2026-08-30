# ML-04-KNN

โปรเจกต์นี้เป็นการทดลองใช้ **K-Nearest Neighbors (KNN)** สำหรับจำแนกประเภทเกียร์ของรถยนต์จากข้อมูลใน `car data.csv`

## Dataset

ไฟล์ข้อมูลอยู่ที่

```text
data-cars/car data.csv
```

Features ที่ใช้

| Feature | ความหมาย |
|---|---|
| Year | ปีของรถ |
| Selling_Price | ราคาขาย |
| Present_Price | ราคาปัจจุบัน |
| Kms_Driven | ระยะทางที่รถวิ่ง |
| Owner | จำนวนเจ้าของ |

Target ที่ต้องการทำนายคือ

```text
Transmission
```

มี 2 Class

- Manual
- Automatic

## หลักการทำงานของ KNN

KNN จะทำงานโดย

1. รับข้อมูลรถที่ต้องการทำนาย
2. คำนวณระยะห่างจากข้อมูลใน Training
3. เลือกข้อมูลที่อยู่ใกล้ที่สุดจำนวน `K` ตัว
4. ดูว่า Class ไหนมีจำนวนมากที่สุด
5. ทำนายเป็น Class นั้น

## ขั้นตอนการทำงาน

### STEP 1: Load Data

อ่านข้อมูลจาก `car data.csv` และเลือก Features กับ Target ที่ต้องการใช้

### STEP 2: Preprocess Data

เปลี่ยนค่า `Manual` และ `Automatic` เป็นตัวเลข และแบ่งข้อมูลเป็น

- Training Data 70%
- Validation Data 15%
- Test Data 15%

จากนั้นใช้ `StandardScaler` เพื่อปรับ Scale ของข้อมูลก่อนใช้ KNN

### STEP 3: Find Best K

ทดลองค่า K หลายค่า

```text
1, 2, 3, 5, 10, 11, 15, 21, 30
```

แล้วเปรียบเทียบ Validation Accuracy เพื่อหาค่า K ที่เหมาะสม

จากผลการทดลอง ค่า K ประมาณ **10–11** ให้ Accuracy สูงที่สุดประมาณ **95.6%**

### STEP 4: Test Model

นำค่า K ที่เลือกมาใช้กับ Test Data และดูผลด้วย

- Accuracy
- Classification Report
- Confusion Matrix

### STEP 5: Compare with Baseline

เปรียบเทียบผลของ KNN กับ Baseline ซึ่งเป็นการทาย Class ที่พบมากที่สุดตลอด

ถ้า KNN มี Accuracy มากกว่า Baseline แสดงว่า KNN ทำได้ดีกว่าวิธีพื้นฐาน

## Output

ผลลัพธ์จะถูกเก็บไว้ใน

```text
classification/outputs/
```

ได้แก่

```text
01_k_curve.png
02_confusion_matrix.png
predictions.csv
```

- `01_k_curve.png` แสดง Accuracy ของแต่ละค่า K
- `02_confusion_matrix.png` แสดงผลการทำนายถูกและผิด
- `predictions.csv` เก็บผลทำนายของ Test Data

## ผลการทดลอง

จากผลลัพธ์ที่ได้

```text
Test Data = 46 samples
Accuracy ≈ 86.96%
Best K ≈ 10–11
Validation Accuracy ≈ 95.6%
```

## สรุป

โปรเจกต์นี้เป็นการทดลองใช้ KNN กับข้อมูลรถยนต์ โดยทดลองค่า K หลายค่า แล้วเลือกค่า K ที่ให้ผลดีที่สุดจาก Validation Data จากนั้นนำไปทดสอบกับ Test Data และเปรียบเทียบผลกับ Baseline และ Scikit-learn
