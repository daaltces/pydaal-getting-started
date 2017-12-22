from Kmeans import Kmeans
from daal.data_management import HomogenNumericTable
from utils import printNumericTable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Create train and test datasets
data = load_iris()
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=42)
trainData = HomogenNumericTable(x_train)
testData=HomogenNumericTable(x_test)
nD_y_test = [[x] for x in y_test]
testGroundTruth = HomogenNumericTable(nD_y_test)

'''
Instantiate Kmeans object Kmeans(nClusters, maxIterations=300, initialCentroidMethod = 'plusPlusDense',
                 method = 'dense', oversamplingFactor =0.5, nRounds=5,
                 accuracyThreshold = 0.0001, gamma = 1.0, distanceType = 'euclidean',
                 assignFlag = True, dtype = float64)
'''
daal_kmeans = Kmeans(3, assignFlag = True) #no. of clusters
#Train
trainingResult = daal_kmeans.compute(trainData)
#Cluster centroids, assignments and objective function
centroids = trainingResult.centroidResults
labels = trainingResult.clusterAssignments
objFunction= trainingResult.objectiveFunction
#Predict
predictResults = daal_kmeans.predict(centroids,testData)

#Serialize centroids [optional]
daal_kmeans.serialize(centroids, fileName='kmeans', useCompression=True)
#Deserialize centroids [optional]
centroids = daal_kmeans.deserialize(fileName='kmeans.npy', useCompression=True)
#Print predicted responses and actual responses
printNumericTable(centroids, "Kmeans Centroids ")



