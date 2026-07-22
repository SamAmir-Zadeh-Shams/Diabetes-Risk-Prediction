from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler

window = Tk()
window.geometry("550x500")

window.title("Diabetes Predictor")

frame = Frame(window)
frame.grid(row=0,column=0)

title = Label(frame, text = "Diabetes Predictor")
title.grid(row=0, column = 0)

def input_control_age(event):
  if event.keysym in ("BackSpace", "Delete", "Left", "Right"):
    return
  
  if event.char.isdigit():
    return

  # if event.char == ".":

  #   previousText = event.widget.get()


  #   if "." in previousText:
  #     return "break"
  #   return
  
  return "break"

def input_control(event):
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


ageTitle = Label(frame, text = "Age")
ageTitle.grid(row=1,column=0)

age = Entry(frame, width=5, font=("Helvetica", 24),justify = 'center', bg = "white", fg = "black")
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

heart_diseaseTitle = Label(frame, text = "Heart Disease")
heart_diseaseTitle.grid(row=4, column = 0)

options_heart_disease = ["--","True","False"]

defaultOption2  = StringVar()
defaultOption2.set(options_heart_disease[0])

heart_disease = OptionMenu(frame, defaultOption2, *options_heart_disease)
heart_disease.grid(row=4, column=1)

#---------------------------------------------------------------------

smoking_historyTitle = Label(frame, text = "Smoking History")
smoking_historyTitle.grid(row=5, column = 0)

options_smoking_history = ["--","No Info","never","former","current","not current"]

defaultOption3  = StringVar()
defaultOption3.set(options_smoking_history[0])

smoking_history = OptionMenu(frame, defaultOption3, *options_smoking_history)
smoking_history.grid(row=5, column=1)

#---------------------------------------------------------------------

bmiTitle = Label(frame, text = "BMI")
bmiTitle.grid(row=6,column=0)

bmi = Entry(frame, width=5, font=("Helvetica", 24),justify = 'center', bg = "white", fg = "black")
bmi.grid(row=6,column=1)
bmi.bind("<KeyPress>", input_control)

#---------------------------------------------------------------------

HbA1c_levelTitle = Label(frame, text = "HbA1c Level")
HbA1c_levelTitle.grid(row=7,column=0)

HbA1c_level = Entry(frame, width=5, font=("Helvetica", 24),justify = 'center', bg = "white", fg = "black")
HbA1c_level.grid(row=7,column=1)
HbA1c_level.bind("<KeyPress>", input_control)

#---------------------------------------------------------------------

blood_glucose_levelTitle = Label(frame, text = "Blood Glucose Level")
blood_glucose_levelTitle.grid(row=8,column=0)

blood_glucose_level = Entry(frame, width=5, font=("Helvetica", 24),justify = 'center', bg = "white", fg = "black")
blood_glucose_level.grid(row=8,column=1)
blood_glucose_level.bind("<KeyPress>", input_control)


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

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")

# # Confusion matrix visual
# import matplotlib.pyplot as plt
# import seaborn as sns

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Not Diabetic', 'Diabetic'],
#             yticklabels=['Not Diabetic', 'Diabetic'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.savefig('confusion_matrix.png')
# plt.show()

# # Bar graph of metrics
# metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
# values = [accuracy, precision, recall, f1]
# plt.figure(figsize=(6,4))
# plt.bar(metrics, values, color=['steelblue', 'orange', 'green', 'red'])
# plt.ylim(0, 1)
# plt.title('Model Performance Metrics')
# plt.ylabel('Score')
# plt.tight_layout()
# plt.savefig('metrics_bar.png')
# plt.show()


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

def submitButton():

  current_age = age.get()
  current_gender = defaultOption.get()
  current_hypertension = defaultOption1.get()
  current_heart_disease = defaultOption2.get()
  current_smoking_history = defaultOption3.get()
  current_bmi = bmi.get()
  current_hba1c = HbA1c_level.get()
  current_blood_glucose = blood_glucose_level.get()

  if current_gender != "--" and current_age != "" and current_hypertension != "--" and current_heart_disease != "--" and current_bmi != "" and current_smoking_history != "--" and current_hba1c != "" and current_blood_glucose != "":
    hyper_num = 1 if current_hypertension == "True" else 0
    heart_num = 1 if current_heart_disease == "True" else 0 

    age_num = float(current_age)
    bmi_num = float(current_bmi)
    HbA1c_level_num = float(current_hba1c)
    blood_glucose_num = float(current_blood_glucose)
    
    result = prediction(age_num,current_gender,hyper_num,heart_num,current_smoking_history,bmi_num,HbA1c_level_num, blood_glucose_num)

    resultInt = int(result[0])
    print(resultInt)
    if resultInt == 0:
      submitLabelMessage.set("You do not have diabetes")
    else:
      submitLabelMessage.set("You do have diabetes")







submit = Button(frame, text = "Submit", state = NORMAL, command = lambda: submitButton())
submit.grid(row=9,column=0)

submitLabelMessage = StringVar()

submitLabel = Label(frame, textvariable = submitLabelMessage)
submitLabel.grid(row=10,column=0)



if (prediction(31,"Male",0,0,"never",27.8,5.6,98)) == 1:
  pass
else:
  pass
# print(y.value_counts())

window.mainloop()