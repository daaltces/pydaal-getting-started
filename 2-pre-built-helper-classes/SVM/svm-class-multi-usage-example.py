from SVM import MultiSVM
from daal.data_management import HomogenNumericTable
from utils import printNumericTables
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split

#create train and test dataset
data = load_digits()
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=42)
trainData = HomogenNumericTable(x_train)
z = [[x] for x in y_train]
trainDependentVariables= HomogenNumericTable(z)
z = [[x] for x in y_test]
testData=HomogenNumericTable(x_test)
testGroundTruth = HomogenNumericTable(z)

'''
Instantiate SVM object MultiSVM(nClasses, method="boser", C = 1, tolerence = 0.001, tau = 0.000001, maxIterations = 1000000, cacheSize = 8000000, doShrinking = True, kernel = 'linear',
                 sigma = 0,k=1, b=0,dtype=float64)
'''
daal_svm = MultiSVM(10,cacheSize=600000000)
#Training
trainingResult = daal_svm.training(trainData,trainDependentVariables)
#Predict
predictResponses = daal_svm.predict(trainingResult, testData)
#Serialize
daal_svm.serialize(trainingResult, fileName='svm')
#deserialize
dese_trainingRes = daal_svm.deserialize(fileName='svm.npy')
#or predict with quality metrics
predictResponses, metrics = daal_svm.predictWithQualityMetrics(dese_trainingRes,testData, testGroundTruth)
daal_svm.printAllQualityMetrics(metrics)


printNumericTables (
    testGroundTruth, predictResponses,
    "Ground truth\t", "Classification results",
    "SVN classification results (first 20 observations):", 20 , flt64=False
)

