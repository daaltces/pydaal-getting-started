from SVM import BinarySVM
from daal.data_management import HomogenNumericTable
from utils import printNumericTables
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
nD_y_train= [[x] for x in y_train]
trainDependentVariables= HomogenNumericTable(nD_y_train)
nD_y_test = [[x] for x in y_test]
testGroundTruth = HomogenNumericTable(nD_y_test)

'''
Instantiate SVM object BinarySVM(nClasses, method="boser", C = 1, tolerence = 0.001, tau = 0.000001, maxIterations = 1000000, cacheSize = 8000000, doShrinking = True, kernel = 'linear',
                 sigma = 0,k=1, b=0,dtype=float64)
'''
daal_svm = BinarySVM(cacheSize=6000000)
#Train
trainingResult = daal_svm.training(trainData,trainDependentVariables)
#Serialize
daal_svm.serialize(trainingResult, fileName='svm', useCompression=True)
#Deserialize
dese_trainingRes = daal_svm.deserialize(fileName='svm.npy', useCompression=True)
#Predict
predictResults = daal_svm.predict(trainingResult,testData)
#or Predict and calculate metrics
predictResults, metrics = daal_svm.predictWithQualityMetrics(trainingResult,testData, testGroundTruth)

#Print quality metrics
daal_svm.printAllQualityMetrics(metrics)

#Print predicted responses and actual responses
printNumericTables (
    testGroundTruth, predictResults,
    "Ground trutht", "Classification results",
    "SVN classification results (first 20 observations):", 20,  flt64=False
)



