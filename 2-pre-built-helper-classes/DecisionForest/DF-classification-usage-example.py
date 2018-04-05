import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','python','source'))
import numpy as np
from DecisionForest import Classification
from daal.data_management import HomogenNumericTable
from utils import printNumericTables,printNumericTable
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from  daal.algorithms import decision_forest, classifier


# Create train and test datasets
#***Binary classifier***
print("**** Binary Classifier****")
data = load_digits(n_class=2)
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=42)
trainData = HomogenNumericTable(x_train)
testData=HomogenNumericTable(x_test)
nD_y_train=  y_train[:,np.newaxis]
trainDependentVariables= HomogenNumericTable(nD_y_train)
nD_y_test =  y_test[:,np.newaxis]
testGroundTruth = HomogenNumericTable(nD_y_test)

'''
Instantiate Decision Forest object: Classification(nClasses, nTrees = 100, observationsPerTreeFraction = 1,featuresPerNode=0,maxTreeDepth=0,
				 minObservationsInLeafNodes=1,impurityThreshold=0,varImportance='MDI')
'''
daal_DF = Classification(len(np.unique(y)),observationsPerTreeFraction=0.7,varImportance='MDI',resultsToCompute=3)
#Train
trainingResult = daal_DF.training(trainData,trainDependentVariables)
#Predict
predictResults = daal_DF.predict(trainingResult,testData)
#Evaluate you model
qualityMet = daal_DF.qualityMetrics(predictResults,testGroundTruth)
#print accuracy
print("Accuracy".format(qualityMet.get('accuracy')))
#print confusion matrix
printNumericTable(qualityMet.get('confusionMatrix'),"Confusion Matrix")
#print all metrics
print("All available metrics")
daal_DF.printAllQualityMetrics(qualityMet)
#Serialize and save
daal_DF.serialize(trainingResult, fileName='DF', useCompression=True)
#Deserialize
dese_trainingRes = daal_DF.deserialize(fileName='DF.npy', useCompression=True)

#Print predicted responses and actual responses
printNumericTables (
    testGroundTruth, predictResults,
    "Ground truth", "Classification results",
    "Decision Forest classification results (first 20 observations):", 20,  flt64=False
 )

#*****Multi-classifier
print("**** Multi-Classifier****")
data = load_digits()
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=42)
trainData = HomogenNumericTable(x_train)
testData=HomogenNumericTable(x_test)
nD_y_train=  y_train[:,np.newaxis]
trainDependentVariables= HomogenNumericTable(nD_y_train)
nD_y_test =  y_test[:,np.newaxis]
testGroundTruth = HomogenNumericTable(nD_y_test)

'''
Instantiate Decision Forest object Classification(nClasses, nTrees = 100, observationsPerTreeFraction = 1,featuresPerNode=0,maxTreeDepth=0,
				 minObservationsInLeafNodes=1,impurityThreshold=0,varImportance='MDI')
'''
daal_DF = Classification(len(np.unique(y)),observationsPerTreeFraction=0.7)
#Train
trainingResult = daal_DF.training(trainData,trainDependentVariables)
#Predict
predictResults = daal_DF.predict(trainingResult,testData)
#Evaluate you model
qualityMet = daal_DF.qualityMetrics(predictResults,testGroundTruth)
#print accuracy
print("Accuracy".format(qualityMet.get('averageAccuracy')))
#print confusion matrix
printNumericTable(qualityMet.get('confusionMatrix'),"Confusion Matrix")
#print all metrics
print("All available metrics")
daal_DF.printAllQualityMetrics(qualityMet)
#Serialize and save
daal_DF.serialize(trainingResult, fileName='DF', useCompression=True)
#Deserialize
dese_trainingRes = daal_DF.deserialize(fileName='DF.npy', useCompression=True)

#Print predicted responses and actual responses
printNumericTables (
    testGroundTruth, predictResults,
    "Ground truth", "Classification results",
    "Decision Forest classification results (first 20 observations):", 20,  flt64=False
 )



