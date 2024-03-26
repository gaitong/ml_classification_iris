# สอน Machine Learning - Classification

แบ่งกลุ่มข้อมูลจาก IRIS dataset : setosa, virginica, versicolor
### ด้วย skLearn model DecisionTreeClassifier & SVM

Installation lib
```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
```
#### 1. ขั้นเตรียมข้อมูล
โหลดข้อมูล csv จาก dataset ตัวอย่าง
```python
df = sns.load_dataset('iris')
```
แสดงข้อมูลตัวอย่าง ด้วยคำสั่ง และคุณสมบัติต่าง ๆ
```python
df #ตัวอย่างข้อมูล
df.dtypes #รายละเอียดข้อมูล
df.shape #ขนาดข้อมูล
df.columns #แสดงชื่อคอลัมน์ทั้งหมด
```
แสดงค่าข้อมูลบนกราฟ scatter
```python
sns.scatterplot(x='sepal_length',y='sepal_width',data=df)
```
กำหนดการแสดงค่าตามประเภท species ด้วยการใส่ค่าตัวแปร hue='species'
```python
sns.scatterplot(x='sepal_length',y='sepal_width',hue='species',data=df)
```
ตัวอย่างกราฟที่ได้
![1](/media/1-form.png)
ทดลองวาดกราฟ pairplot ของข้อมูลจากคู่ความสัมพันธ์
```python
sns.pairplot(df,hue='species')
```
เตรียมข้อมูล X,y ก่อนเตรียมสามารถใช้คำสั่งเช็ค null
```python
df.isnull().sum()
```
แบ่งข้อมูล output ที่ต้องการเทรน ด้วยค่า y เท่ากับ คอลัมน์ species
```python
y = df[['species']]
```
แบ่งข้อมูล feature ที่ต้องการเทรน ด้วยค่า X ที่ไม่เท่ากับ คอลัมน์ species
```python
X = df.drop('species',axis=1)
```
#### 2.ขั้นเทรน
เตรียมแบ่งข้อมูล Feature และ Label ด้วยการ import lib ตามด้านล่างนี้
```python
from sklearn.model_selection import train_test_split
```
กำหนดค่าตัวแปร temp = 0.2 เพื่อทำการแบ่งข้อมูล Train 80% : Test 20%
```python
temp = .2
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=temp,random_state=1)
```
แสดงค่าขนาดข้อมูล X,y ก่อนเทรน ด้วยคำสั่ง
title
```python
X_train.shape
X_test.shape
```
### แบบที่ 1 Model DecisionTreeClassifier เพื่อเริ่มเทรนข้อมูล
```python
#train model decition tree
from sklearn import tree
model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)
```
เทส score จากข้อมูลการเทรน
```python
model.score(X_train, y_train)
```
เทส score จากข้อมูลเทส ที่ machine ยังไม่เคยเห็น
```python
model.score(X_test, y_test)
```
#### 3.ขั้นทำนาย
ทดลองใส่ค่า feature เอง
```python
y_prediction = model.predict([[6.7,3.0,5.2,2.3]])
y_prediction
```
ตัวอย่างผลลัพธ์ที่ได้ \
array(['virginica'], dtype=object)
#### ทำนายจากค่า X_test
```python
y_prediction = model.predict(X_test)
y_prediction
```
ตัวอย่างผลการทำนาย \
array(['setosa', 'versicolor', 'versicolor', 'setosa', 'virginica',
       'versicolor', 'virginica', 'setosa', 'setosa', 'virginica',
       'versicolor', 'setosa', 'virginica', 'versicolor', 'versicolor',
       'setosa', 'versicolor', 'versicolor', 'setosa', 'setosa',
       'versicolor', 'versicolor', 'virginica', 'setosa', 'virginica',
       'versicolor', 'setosa', 'setosa', 'versicolor', 'virginica'],
      dtype=object)
#### วาดกราฟ tree ด้วยคำสั่ง
```python
tree.plot_tree(model)
```
### แบบที่ 2 Model SVM เพื่อเริ่มเทรนข้อมูล
```python
#import lib
from sklearn import svm
model = svm.SVC()
#เทรนนิ่ง
model.fit(X_train, y_train
#เทส
model.score(X_train, y_train)
model.score(X_test, y_test)
#ทำนาย
y_prediction = model.predict(X_test)
y_prediction
```
#### ตัวอย่างผลการทำนาย
array(['setosa', 'versicolor', 'versicolor', 'setosa', 'virginica',
       'versicolor', 'virginica', 'setosa', 'setosa', 'virginica',
       'versicolor', 'setosa', 'virginica', 'versicolor', 'versicolor',
       'setosa', 'versicolor', 'versicolor', 'setosa', 'setosa',
       'versicolor', 'versicolor', 'virginica', 'setosa', 'virginica',
       'versicolor', 'setosa', 'setosa', 'versicolor', 'virginica'],
      dtype=object)