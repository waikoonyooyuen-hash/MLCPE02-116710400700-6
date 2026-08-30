# ML-06-NN

โปรเจกต์นี้เป็นการทดลองใช้ **Neural Network (NN)** สำหรับจำแนกรูปตัวเลขที่เขียนด้วยลายมือ

## Dataset

ใช้ **Handwritten Digits Dataset** จาก `scikit-learn`

มีรูปตัวเลข `0-9` จำนวน 1,797 รูป โดยแต่ละรูปมีขนาด 8x8 pixels

## หลักการทำงาน

```text
Dataset
   ↓
Preprocess
   ↓
Split Data
   ↓
Train Neural Network
   ↓
Prediction
   ↓
Evaluation
```

## ขั้นตอนการทำงาน

### Step 1: Load Data
โหลดรูปตัวเลขจาก `scikit-learn`

### Step 2: Preprocessing
ปรับค่าของ Pixel ให้อยู่ในช่วง 0-1

### Step 3: Split Data
แบ่งข้อมูลเป็น Training, Validation และ Test

### Step 4: Train Neural Network
นำ Training Data ไปฝึก Neural Network

### Step 5: Prediction
ใช้โมเดลทำนายว่ารูปเป็นเลขอะไร

### Step 6: Evaluation
วัดผลด้วย Accuracy และ Confusion Matrix

## Output

```text
outputs/
├── nn_model.pkl
├── confusion_matrix.png
├── training_history.png
└── prediction_sample.png
```

## วิธี Run

```bash
pip install -r requirements.txt
python main.py
```

ทดสอบตัวอย่าง Prediction:

```bash
python test_nn.py
```

## สรุป

โปรเจกต์นี้ใช้ Neural Network ในการจำแนกรูปตัวเลข 0-9 โดยให้โมเดลเรียนรู้จาก Training Data แล้วนำไปทำนายข้อมูลใหม่
