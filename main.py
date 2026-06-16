from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

dataset = pd.read_csv("diabetes_prediction_dataset.csv") # create the dataframe using the dataset

# x = dataset.iloc[:, :-1]
# y = dataset.iloc[:,-1]

dataset = pd.get_dummies(dataset) # get data and translate all data that is text into 1s and 0s

y = dataset.pop('diabetes')
x = dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state= 19)

UnderSample = RandomUnderSampler(sampling_strategy=0.3, random_state=19)

X_train_resampled, y_train_resampled = UnderSample.fit_resample(X_train, y_train)

scaler = StandardScaler()
scaler.fit(X_train_resampled)

X_train = scaler.transform(X_train_resampled)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=11, weights = "distance")
classifier.fit(X_train, y_train_resampled)

y_pred = classifier.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

def prediction(age,gender,hypertension,heart_disease,smoking_history,bmi,HbA1c,glucose):
  input_data = pd.DataFrame({
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'bmi': [bmi],
    'HbA1c_level':[HbA1c],
    'blood_glucose_level':[glucose],
    'gender':[gender],
    'smoking_history': [smoking_history]
  })

  input_data = pd.get_dummies(input_data)

  for i in x.columns:
    if i not in input_data.columns:
      input_data[i] = 0

  input_data = input_data[x.columns]

  input_data_scaled = scaler.transform(input_data)

  return classifier.predict(input_data_scaled)

print(prediction(31,"Male",0,0,"never",27.8,5.6,98))

# print(y.value_counts())