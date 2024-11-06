# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Preprocess Data: Load the dataset, select relevant columns, and map labels to numeric values.

2.Split Data: Divide the dataset into training and test sets.

3.Vectorize Text: Convert text messages into a numerical format using TfidfVectorizer.

4.Train SVM Model: Fit an SVM model with a linear kernel on the training data.

5.Evaluate Model: Predict on the test set and print the accuracy score and classification report.

6.Visualize Results: Plot the confusion matrix, ROC curve, and precision-recall curve for detailed evaluation.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: VAISHALI BALAMURUGAN
RegisterNumber:  212222230164
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
```
data.head()
```
data.tail()
```
```
data.info()
```
```
data.isnull().sum()
```
```
x=data['v2'].values
```
```
y=data['v1'].values
```
```
y.shape
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
y_train.shape
```
```
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
```
```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```
x_train.shape
```
```
type(x_train)
```
```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## output:


![image](https://github.com/user-attachments/assets/2aa966fd-1de6-4c0d-ba3c-07c0335f4ec8)
![image](https://github.com/user-attachments/assets/4f26dafb-1377-4e00-975d-510cd0d7cc77)
![image](https://github.com/user-attachments/assets/ae8abb9e-a0e6-48db-94b8-66470d5437c8)
![image](https://github.com/user-attachments/assets/6c6e37b6-2990-491f-a61d-9d776816ecd2)
![image](https://github.com/user-attachments/assets/1ebd23fa-6bd6-47ad-a8bc-bc15d7272544)
![image](https://github.com/user-attachments/assets/fea9bc08-8f6d-483d-9c90-b65e7d040552)
![image](https://github.com/user-attachments/assets/a8e1d4fa-c35a-45de-a7ec-943c58bd3660)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
