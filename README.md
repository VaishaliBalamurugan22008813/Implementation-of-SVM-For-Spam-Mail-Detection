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
```

<h2>OUTPUT<h2>
![Screenshot 2024-11-06 093420](https://github.com/user-attachments/assets/79efa94e-f891-457a-82d0-10adac89735c)
![Screenshot 2024-11-06 093439](https://github.com/user-attachments/assets/0e53b13f-6ae5-4e1f-95ab-217117b72caa)
![Screenshot 2024-11-06 093453](https://github.com/user-attachments/assets/d11c8db5-5a07-461e-a23c-e874043b7cad)
![Screenshot 2024-11-06 093504](https://github.com/user-attachments/assets/ad93b019-861e-4df7-b69d-4babdf26d096)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
