# Lab 8 - DCNN

## 1. วัตถุประสงค์

Lab นี้เป็นการทดลองใช้ Deep Convolutional Neural Network (DCNN) สำหรับจำแนกภาพออกเป็น 2 ประเภท โดยเปลี่ยน Dataset จากตัวอย่างเดิมของ Lab เป็นภาพข้าว (Rice) และข้าวโพด (Corn)

จุดประสงค์คือให้โมเดลเรียนรู้ลักษณะของภาพจาก Dataset และสามารถทำนายภาพใหม่ได้ว่าเป็น Rice หรือ Corn

## 2. Dataset

ใน Lab นี้ใช้ Dataset 2 classes ได้แก่

- Corn: ภาพข้าวโพด
- Rice: ภาพข้าว

โดยเก็บภาพแยกไว้ในโฟลเดอร์ของแต่ละ class เพื่อให้โปรแกรมสามารถอ่านและกำหนด label ให้กับภาพได้อัตโนมัติ

## 3. โครงสร้างการทำงาน

โปรแกรมทำงานตามลำดับดังนี้

1. โหลดภาพจาก Dataset
2. Preprocess และปรับขนาดภาพ
3. แบ่งข้อมูลสำหรับ Train และ Test
4. สร้างและ Train โมเดล DCNN
5. ทดสอบโมเดลกับข้อมูล Test
6. ประเมินผลการทำงานของโมเดล
7. บันทึกผลลัพธ์ไว้ในโฟลเดอร์ `outputs`

## 4. โครงสร้างไฟล์

```text
ML-08-DCNN
└── mini-proj
    ├── Dataset
    │   ├── Corn
    │   └── Rice
    ├── outputs
    ├── main.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── split_data.py
    ├── dcnn_model.py
    ├── evaluate.py
    └── README.md
```

## 5. วิธีติดตั้ง Library

เปิด Terminal ในโฟลเดอร์โปรเจกต์ แล้วใช้คำสั่ง

```bash
python -m pip install -r requirements.txt
```

แนะนำให้ใช้ Python 3.12 (64-bit)

## 6. วิธี Run

เปิดไฟล์ `main.py` ใน VS Code แล้วกด Run

หรือสามารถใช้คำสั่ง

```bash
python main.py
```

หลังจากโปรแกรมทำงานเสร็จ ผลลัพธ์ต่าง ๆ จะถูกเก็บไว้ในโฟลเดอร์ `outputs`

## 7. ผลลัพธ์

ผลลัพธ์ที่ได้จากการทดลองประกอบด้วยข้อมูลการ Train และการประเมินโมเดล เช่น Accuracy, Loss และ Confusion Matrix รวมถึงไฟล์โมเดลที่ผ่านการ Train แล้ว

ผลลัพธ์เหล่านี้สามารถใช้ดูว่า DCNN สามารถจำแนกภาพ Corn และ Rice ได้ดีแค่ไหน

## 8. สรุป

จากการทดลองได้เรียนรู้การทำงานของ Deep Convolutional Neural Network ตั้งแต่การเตรียม Dataset การ Preprocess ภาพ การ Train โมเดล ไปจนถึงการทดสอบและประเมินผล โดยใน Lab นี้ใช้ภาพ Corn และ Rice เป็น Dataset สำหรับการจำแนก 2 classes
