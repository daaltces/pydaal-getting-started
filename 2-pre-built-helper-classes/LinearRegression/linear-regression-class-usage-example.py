import sys
import numpy as np
sys.path.append(r'..')
from LinearRegression import LinearRegression
from utils import printNumericTable
from daal.data_management import HomogenNumericTable

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

data = load_boston()
X= data.data
Y=data.target
x_train, x_test, y_train_temp, y_test_temp=train_test_split(X,Y,test_size=0.40,random_state=42)
y_train = y_train_temp[:,np.newaxis]
y_test= y_test_temp[:,np.newaxis]

trainData = HomogenNumericTable(x_train)
trainDependentVariables = HomogenNumericTable(y_train)
testData =HomogenNumericTable(x_test)
testGroundTruth =HomogenNumericTable(y_test)

#Instantiate Linear Regression object
lr = LinearRegression()
#Training
trainingResult = lr.training(trainData,trainDependentVariables)
#Prediction
prediction_nT = lr.predict(trainingResult,testData)
#Evaluation
qualityMet = lr.qualityMetrics(trainingResult,prediction_nT,testGroundTruth)
printNumericTable(qualityMet.get('rms'),"Root mean square")
#To print all the metrics
lr.printAllQualityMetrics(qualityMet)
#To predict and evaluate. Note that this method performs predictions on both unrestricted and restricted(reduced) model
predRes, predResRed, qualityMet = lr.predictWithQualityMetrics(trainingResult, testData, testGroundTruth,[1,2])
#Serialize
lr.serialize(trainingResult, fileName = 'LR.npy')
#Deseriailze
de_trainingResult = lr.deserialize(fileName = "LR.npy")
#Print Metrics results
#print predicted responses and actual response
printNumericTable (predRes, "Linear Regression prediction results: (first 10 rows):", 10)
printNumericTable (testGroundTruth, "Ground truth (first 10 rows):", 10)
