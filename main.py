from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv("diabetes_prediction_dataset.csv") # create the dataframe using the dataset

x = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]

dataset = pd.get_dummies(dataset) # get data and translate all data that is text into 1s and 0s

y = dataset.pop('diabetes')
x = dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

def prediction():
  pass