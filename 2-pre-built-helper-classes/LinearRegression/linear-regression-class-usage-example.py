
from LinearRegression import LinearRegression
from utils import printNumericTable
from daal.data_management import HomogenNumericTable
import numpy as np

nFeatures = 10
nDependentVariables = 2

seeded = np.random.RandomState (42)
trainData =HomogenNumericTable(seeded.rand(200,nFeatures))
trainDependentVariables = HomogenNumericTable(seeded.rand (200, nDependentVariables))
testData =HomogenNumericTable(seeded.rand(50,nFeatures))
testGroundTruth = HomogenNumericTable(seeded.rand (50, nDependentVariables))

#Instantiate Linear Regression object
lr = LinearRegression()
#Training
trainingResult = lr.training(trainData,trainDependentVariables)
#Prediction
pred_array = lr.predict(trainingResult,trainData)
#Serialize
lr.serialize(trainingResult, fileName = 'trainingResult.npy')
#Deseriailze
de_trainingResult = lr.deserialize(fileName = "trainingResult.npy")
#Predict with Metrics
predRes, predResRed, singleBeta, groupBeta = lr.predictWithQualityMetrics(trainingResult, trainData, trainDependentVariables, reducedBetaIndex=[2,10])
#Print Metrics results
lr.printAllQualityMetrics(singleBeta,groupBeta)
#print predicted responses and actual response
printNumericTable (predRes, "Linear Regression prediction results: (first 10 rows):", 10)
printNumericTable (predResRed, "Linear Regression prediction results: (first 10 rows):", 10)
printNumericTable (trainDependentVariables, "Ground truth (first 10 rows):", 10)

