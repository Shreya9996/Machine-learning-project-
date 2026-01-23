
import pandas as pd 
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np 
data = pd.read_csv(r"C:\Users\Admin\Documents\Scikit _learn\datasets_4123_6408_framingham.csv")
data.drop("education",axis=1,inplace=True)



numeric_cols = ["cigsPerDay","BPMeds","totChol","BMI","heartRate","glucose"]


for col in numeric_cols:
    mean_value = data[col].mean()
    data[col] = data[col].fillna(mean_value)

from sklearn.model_selection import train_test_split
x = data.drop("TenYearCHD",axis=1)
y = data["TenYearCHD"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

balance_data = pd.concat([x_train,y_train],axis=1)

majority = 
# sample = SMOTE()
# x,y = sample.fit_resample(data.drop("TenYearCHD",axis=1),data["TenYearCHD"])

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# x_train ,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# scaled = StandardScaler()
# x_train_scaled  = scaled.fit_transform(x_train)
# x_test_scaled = scaled.transform(x_test)
# # print(x_train_scaled)

# from sklearn.linear_model import LogisticRegression

# from sklearn.ensemble import RandomForestClassifier
# le = RandomForestClassifier()
# le.fit(x_train_scaled,y_train)
# y_pred = le.predict(x_test_scaled)



# from  sklearn.metrics import accuracy_score,confusion_matrix,classification_report
# print(accuracy_score(y_test,y_pred))


# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))


# single_input = x_train_scaled[25].reshape(1,-1)
# print("predict : ",le.predict(single_input)[0])
# print("acctual :",y_train.iloc[25])

import pickle

pickle.dump(le,open("train_maodel_1","wb"))
pickle.dump(scaled,open("standeer_scaler_1","wb"))

with open("train_maodel_1","rb") as file:
    model = pickle.load(file)  

with open("standeer_scaler_1","rb") as file1:
    scaled = pickle.load(file1)   
    


def predict(model,scaled,male,age,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose):
    encode_male = 1 if male.lower()=="yes" else 0
    encode_BPMeds = 1 if BPMeds.lower()=="yes" else 0
    encode_prevalentStroke = 1 if prevalentStroke.lower()=="yes" else 0
    encode_prevalentHyp =  1 if prevalentHyp.lower() =="yes" else 0 
    encode_diabetes = 1 if diabetes.lower()=="yes" else 0 
    encode_currentSmoker = 1 if currentSmoker.lower()=="yes" else 0
     
    # features = np.array([[encode_male,age,encode_currentSmoker,cigsPerDay,encode_BPMeds,encode_prevalentStroke,encode_prevalentHyp,encode_diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose]])
    # scaler_feature = scaled.transform(features)
    features = pd.DataFrame([[encode_male,age,encode_currentSmoker,cigsPerDay,
                          encode_BPMeds,encode_prevalentStroke,encode_prevalentHyp,
                          encode_diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose]],
                        
                        columns = x_train.columns)

    scaler_feature = scaled.transform(features)

    result = model.predict(scaler_feature)
    return result[0]


male = "no"
age = 61
currentSmoker = "yes"
cigsPerDay = 30
BPMeds = "no"
prevalentStroke = "no"
prevalentHyp = "yes"
diabetes = "no"
totChol = 225
sysBP = 150
diaBP = 95
BMI = 28.58
heartRate = 65
glucose = 103

result = predict(model,scaled,male,age,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose)

if result==1:
    print("This patient has 'Heart Disease !' ")

else :
    print("This patient not have 'Heart Disease !' ")

