import sys, os
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','python','source'))
import numpy as np
from RidgeRegression import RidgeRegression
from utils import printNumericTable
from daal.data_management import HomogenNumericTable

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

data = load_boston()
X= data.data
Y=data.target
x_train, x_test, y_train_temp, y_test_temp=train_test_split(X,Y,test_size=0.40,random_state=42)
y_train = y_train_temp[:,np.newaxis]
y_test=  y_test_temp[:,np.newaxis]

trainData = HomogenNumericTable(x_train)
trainDependentVariables = HomogenNumericTable(y_train)
testData =HomogenNumericTable(x_test)
testGroundTruth =HomogenNumericTable(y_test)
#Instantiate Linear Regression object
rigde = RidgeRegression(ridgeParameters=0.0005)
#Training
trainingResult = rigde.training(trainData,trainDependentVariables)
#Prediction
pred_nT = rigde.predict(trainingResult,trainData)
#Serialize
rigde.serialize(trainingResult, fileName = 'RR.npy')
#Deseriailze
de_trainingResult = rigde.deserialize(fileName = "RR.npy")
#print predicted responses and actual response
printNumericTable (pred_nT, "Ridge Regression prediction results: (first 10 rows):", 10)
printNumericTable (testGroundTruth, "Ground truth (first 10 rows):", 10)