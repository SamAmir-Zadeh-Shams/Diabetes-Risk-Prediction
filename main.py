from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

window = Tk()
window.geometry("550x500")

window.title("Diabetes Predictor")

frame = Frame(window)
frame.grid(row=0,column=0)

title = Label(frame, text = "Diabetes Predictor")
title.grid(row=0, column = 0)

ageTitle = Label(frame, text = "Age")
ageTitle.grid(row=1,column=0)

def input_control_age(event):
  if event.keysym in ("BackSpace", "Delete", "Left", "Right"):
    return
  
  if event.char.isdigit():
    return

  if event.char == ".":

    previousText = event.widget.get()


    if "." in previousText:
      return "break"
    return
  
  return "break"

age = Entry(frame, width=3, font=("Helvetica", 24),justify = 'center', bg = "white", fg = "black")
age.grid(row=1,column=1)
age.bind("<KeyPress>", input_control_age)

#---------------------------------------------------------------------

genderTitle = Label(frame, text = "Gender")
genderTitle.grid(row=2, column = 0)

options_gender = ["--","Male","Female"]

defaultOption  = StringVar()
defaultOption.set(options_gender[0])

gender = OptionMenu(frame, defaultOption, *options_gender)
gender.grid(row=2, column=1)

#---------------------------------------------------------------------

hypertensionTitle = Label(frame, text = "Hypertension")
hypertensionTitle.grid(row=3, column = 0)

options_hypertension = ["--","True","False"]

defaultOption1  = StringVar()
defaultOption1.set(options_hypertension[0])

hypertension = OptionMenu(frame, defaultOption1, *options_hypertension)
hypertension.grid(row=3, column=1)

#---------------------------------------------------------------------



dataset = pd.read_csv("diabetes_prediction_dataset.csv") # create the dataframe using the dataset

# x = dataset.iloc[:, :-1]
# y = dataset.iloc[:,-1]

dataset = pd.get_dummies(dataset) # get data and translate all data that is text into 1s and 0s

y = dataset.pop('diabetes') # have the y dataframe only include the 'diabetes' column
x = dataset # have the x dataframe include all columns except 'diabetes' column

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state= 19) # have X_train be 80% of the x dataframe, y_train be 80% of the y database, X_test be 20% of the x database, and y_test be 20% of the y database, use stratify to preserve the original balance of the dataset, and set random_state equal to a number so it uses the same split in the dataset every time

UnderSample = RandomUnderSampler(sampling_strategy=0.3, random_state=19) # create a random under sampler object that will reduce the number of majority class samples so that the minority will eb about 30% of the majority and use random_state to make sure that the same rows are selected

X_train_resampled, y_train_resampled = UnderSample.fit_resample(X_train, y_train) # have X_train_resampled contains the new training features after removing some majority-class samples and have y_train_resampled contains the corresponding target values

scaler = StandardScaler()
scaler.fit(X_train_resampled) # make calculations using the new resampled training data

X_train = scaler.transform(X_train_resampled) # standardize the resampled training data
X_test = scaler.transform(X_test) # standarize the test data

classifier = KNeighborsClassifier(n_neighbors=11, weights = "distance") #look at the 11 closest trainiing samples and use distance as weights so that closer neighbors have more of an influence on prediction
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

#print(prediction(31,"Male",0,0,"never",27.8,5.6,98))

# print(y.value_counts())

window.mainloop()