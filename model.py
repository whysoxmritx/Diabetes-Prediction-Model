import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# Data Collection and analysis
# PIMA(Prime Indian Diabetes Dataset)
# loading the diabetes dataset to a pandas Dataframe
diabetes_dataset=pd.read_csv('diabetes.csv')
# 0-->Non-diabetic
# 1--> Diabetic 
# separating labels and data
X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']
# Data Standarization 
scalar=StandardScaler()
scalar.fit(X)
standardized_data=scalar.transform(X)
# can also use scalar.fit_transform
# print(standardized_data)
X=standardized_data
Y=diabetes_dataset['Outcome']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
#accuracy score on the training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy score of the training data",training_data_accuracy)
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy Test data: ",test_data_accuracy)
#making a prediction system 
input_data=(3,158,76,36,245,31.6,0.851,28)
#change data to numpy array
input_data_as_numpy_array=np.asarray(input_data)
# reshape the array for one instance 
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
# standarize the input data
std_data=scalar.transform(input_data_reshaped)
print(std_data)
prediction=classifier.predict(std_data)
print(prediction)
if(prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")