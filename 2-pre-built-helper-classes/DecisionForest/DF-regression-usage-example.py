import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','python','source'))
from DecisionForest import Regression
from utils import printNumericTable
from daal.data_management import HomogenNumericTable
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

data = load_boston()
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=42)

trainData = HomogenNumericTable(x_train)
testData=HomogenNumericTable(x_test)
nD_y_train= y_train[:,np.newaxis]
trainDependentVariables= HomogenNumericTable(nD_y_train)
nD_y_test = y_test[:,np.newaxis]
testGroundTruth = HomogenNumericTable(nD_y_test)
'''
Instantiate Decision Forest object Regression(nTrees = 100, observationsPerTreeFraction = 1,featuresPerNode=0,maxTreeDepth=0,
				 minObservationsInLeafNodes=5,impurityThreshold=0,varImportance=None,resultsToCompute=0)
'''
#Instantiate Linear Regression object
daal_DF = Regression(nTrees=100,maxTreeDepth=15,resultsToCompute=3)
#Training
trainingResult = daal_DF.training(trainData,trainDependentVariables)
#Prediction
pred_nT = daal_DF.predict(trainingResult,trainData)
#Serialize the training object
daal_DF.serialize(trainingResult, fileName = 'DF_Reg.npy')
#Deseriailze
de_trainingResult = daal_DF.deserialize(fileName = "DF_Reg.npy")
#print predicted responses and actual response
printNumericTable (pred_nT, "Linear Regression prediction results: (first 10 rows):", 10)
printNumericTable (trainDependentVariables, "Ground truth (first 10 rows):", 10)




