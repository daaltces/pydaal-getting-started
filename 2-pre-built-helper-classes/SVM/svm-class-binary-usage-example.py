import sys
import os
sys.path.append(r'..')
sys.path.append(os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','python','source'))
import numpy as np
from SVM import BinarySVM
from daal.data_management import HomogenNumericTable
from utils import printNumericTables, printNumericTable
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Create train and test datasets
data = load_breast_cancer()
x = data.data
y = data.target
y[y==0]=-1 # DAAL's SVM binary classifier labels must be -1 and 1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=42)
trainData = HomogenNumericTable(x_train)
testData=HomogenNumericTable(x_test)
nD_y_train= y_train[:,np.newaxis]
trainDependentVariables= HomogenNumericTable(nD_y_train)
nD_y_test = y_test[:,np.newaxis]
testGroundTruth = HomogenNumericTable(nD_y_test)

'''
Instantiate SVM object BinarySVM(nClasses, method="boser", C = 1, tolerence = 0.001, tau = 0.000001, maxIterations = 1000000, cacheSize = 8000000, doShrinking = True, kernel = 'linear',
				 sigma = 0,k=1, b=0,dtype=float64)
'''
daal_svm = BinarySVM(cacheSize=6000000)
#Train
trainingResult = daal_svm.training(trainData,trainDependentVariables)
#Predict
predictResults = daal_svm.predict(trainingResult,testData)
#Evaluate you model
qualityMet = daal_svm.qualityMetrics(predictResults,testGroundTruth)
#print accuracy
print(qualityMet.get('accuracy'))
#print confusion matrix
printNumericTable(qualityMet.get('confusionMatrix'))
#print all metrics
daal_svm.printAllQualityMetrics(qualityMet)
#Serialize
daal_svm.serialize(trainingResult, fileName='svm', useCompression=True)
#Deserialize
dese_trainingRes = daal_svm.deserialize(fileName='svm.npy', useCompression=True)

#Print predicted responses and actual responses
printNumericTables (
	testGroundTruth, predictResults,
	"Ground truth", "Classification results",
	"SVM classification results (first 20 observations):", 20,  flt64=False
)



