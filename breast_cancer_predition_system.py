import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\Admin\Documents\Scikit _learn\breast-cancer.csv")
# print(data.head)
# print(data.isnull().sum())
# print(data.duplicated().sum())
# print(data.info())
# print(data.describe())

data = data.drop("id",axis=1)
data["diagnosis"] = data["diagnosis"].map({"M":1,"B":0})

from sklearn.model_selection import train_test_split
x = data.drop("diagnosis",axis=1)
y = data["diagnosis"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print("accuracy score : ",accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix,classification_report
print("confusion_matrix :",confusion_matrix(y_test,y_pred))
print("Classification matrics : ",classification_report(y_test,y_pred))

print(x_train[103])

import pickle
pickle.dump(model,open("model.pkl","wb"))




