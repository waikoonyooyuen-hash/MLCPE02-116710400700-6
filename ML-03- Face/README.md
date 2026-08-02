# ML03_HW2
ชื่อ ไวกูณฐ์ อยู่ยืน 116710400700-6 Sec2

#  การเตรียมข้อมูล (Data Preparation)
Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('faces.csv')

df['face_width'] = df['x1'] - df['x0']
df['face_height'] = df['y1'] - df['y0']
df['face_area'] = df['face_width'] * df['face_height']

np.random.seed(42)
df['age'] = 20 + (df['face_area'] / df['face_area'].max()) * 40 + np.random.normal(0, 5, size=len(df))

อธิบาย code
  1. pd.read_csv('faces.csv'): ดึงข้อมูลจากไฟล์ CSV เข้ามาเก็บในตัวแปร df
  2. df['face_width'] และ df['face_height']: คำนวณหาความกว้างและความสูงของใบหน้า
  3. df['face_area']: นำความกว้างมาคูณความสูงเพื่อหาพื้นที่ใบหน้า
  4. df['age']: สุ่มสร้างค่าอายุสมมติโดยอิงจากขนาดพื้นที่ใบหน้า เพื่อใช้ทำนายในโจทย์ถัดไป

โดยใบหน้านั้นเราได้นำมาข้อมูลมากจาก kaggle

# Lab1 การทำทำนายอายุ (Regression)

Lab นี้ทดลองทำนายอายุด้วยโมเดล 3 แบบ ได้แก่ Simple Linear Regression, Multiple Linear Regression และ PCA Linear Regression
Python
   # --- Part 1: Simple Linear Regression (ใช้ 1 ตัวแปร) ---
X_simple = df[['face_area']]
y = df['age']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)
y_pred_s = simple_model.predict(X_test_s)

   # --- Part 2: Multiple Linear Regression (ใช้หลายตัวแปร) ---
X_multi = df[['width', 'height', 'face_width', 'face_height', 'face_area']]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)
y_pred_m = multi_model.predict(X_test_m)

    # --- Part 3: PCA Linear Regression (ลดมิติข้อมูลก่อนเทรน) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_multi)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_pca, y, test_size=0.2, random_state=42)

pca_model = LinearRegression()
pca_model.fit(X_train_p, y_train_p)
y_pred_p = pca_model.predict(X_test_p)

 อธิบาย code
   1. train_test_split(...): แบ่งข้อมูลเป็นชุดเรียนรู้ (Train 80%) และชุดทดสอบ (Test 20%)
   2. LinearRegression() และ .fit(...): สร้างโมเดลแล้วป้อนข้อมูลให้โมเดลฝึกเรียนรู้
   3. .predict(...): สั่งให้โมเดลทำนายค่าออกมา
   4. StandardScaler() และ PCA(n_components=2): ปรับมาตรฐานข้อมูลและบีบอัดข้อมูลจากหลายตัวแปรให้เหลือ 2 แกนหลัก (PC1, PC2)

# LAB2 การจำแนกประเภทเพศ (Classification)

Lab นี้เปลี่ยนมาทำนายประเภทข้อมูล (Male/Female) โดยใช้ Logistic Regression

    # 1. จำลองข้อมูลเพศ (0 = ชาย, 1 = หญิง)
df['gender'] = (df['face_width'] / df['face_height'] > 1.0).astype(int)
y_gender = df['gender']

    # 2. แบ่งข้อมูล Train / Test
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_pca, y_gender, test_size=0.2, random_state=42)

    # 3. เทรนโมเดล Logistic Regression
clf = LogisticRegression()
clf.fit(X_train_c, y_train_c)

    # 4. ทำนายผล
y_pred_c = clf.predict(X_test_c)
y_prob_c = clf.predict_proba(X_test_c)[:, 1]

 อธิบาย code
   1. (df['face_width'] / df['face_height'] > 1.0): สร้างเงื่อนไขจำลองเพศจากสัดส่วนใบหน้า
   2. LogisticRegression(): สร้างโมเดลจำแนกประเภท
   3. .predict_proba(...): หาค่าความน่าจะเป็นของการเป็นแต่ละคลาส เพื่อเอาไปใช้วาดกราฟ ROC Curve

# LAB3 สรุปผลและการเปรียบเทียบโมเดล (Model Comparison)

ในส่วนนี้เป็นการนำผลการทดลองมาแสดงผ่านกราฟ ตาราง และตัวชี้วัดต่างๆ เช่น $R^2$, MSE, Accuracy, F1-Score
Python
# เปรียบเทียบผล Regression
reg_comp_df = pd.DataFrame({
    'Model': ['Simple Linear Regression', 'Multiple Linear Regression', 'PCA Linear Regression'],
    'Train R²': [r2_score(y_train_s, simple_model.predict(X_train_s)), 
                r2_score(y_train_m, multi_model.predict(X_train_m)), 
                r2_score(y_train_p, pca_model.predict(X_train_p))],
    'Test R²': [r2_score(y_test_s, y_pred_s), r2_score(y_test_m, y_pred_m), r2_score(y_test_p, y_pred_p)],
    'Test MSE': [mean_squared_error(y_test_s, y_pred_s), mean_squared_error(y_test_m, y_pred_m), mean_squared_error(y_test_p, y_pred_p)]
})

print(reg_comp_df)

 อธิบาย code
   1.รวบรวมค่า $R^2$ และ MSE มาสร้างตารางเพื่อเปรียบเทียบดูว่าโมเดลไหนแม่นยำที่สุด และดูว่าเกิด Overfitting หรือไม่ 

 # สรุปผลการทดลอง
 - การทำนายอายุ (Regression): โมเดลที่ใช้หลายตัวแปร (Multiple LR) ให้ผลลัพธ์ดีกว่าการใช้แค่ตัวแปรเดียว (Simple LR) ส่วนโมเดลที่ใช้ PCA สามารถบีบข้อมูลเหลือ 2 Dimensions แต่ยังคงทำนายผลได้ใกล้เคียงกับข้อมูลชุดเต็ม
 - การจำแนกเพศ (Classification): โมเดล Logistic Regression สามารถแบ่งเส้น Decision Boundary เพื่อแยกเพศชายและหญิงได้ดี
 - การประเมินผล: ค่าประสิทธิภาพระหว่าง Train และ Test ใกล้เคียงกัน แสดงว่าโมเดลไม่มีปัญหา Overfitting สามารถนำไปใช้งานจริงได้ครับ
