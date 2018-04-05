
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','python','source'))
from PCA import PCA
from customUtils import getNumericTableFromCSV
from utils import printNumericTable
from sklearn.datasets import load_iris
from daal.data_management import HomogenNumericTable_Float64

iris = load_iris()
X = iris.data
'''
	Instantiate a PCA object :PCA(method = 'defaultDense',covariance = "defaultDense", normalization="zscore", nComponents = 0,isDeterministic = False,fptype=float64, resultsToCompute = None):

'''
pca = PCA(resultsToCompute=3)
X = HomogenNumericTable_Float64(X)
#Compute principle components
pcaResult = pca.compute(X)
res = pca.getResults(pcaResult)

printNumericTable(res["eigenvalues"], "Eigenvalues")
printNumericTable(res["eigenvectors"], "Eigenvectors")
printNumericTable(res["dataForTransform"], "DataForTransform")
printNumericTable(res["means"], "Means")
printNumericTable(res["variances"], "Variances")

#Evaluation
qualityMet=pca.qualityMetrics(pcaResult)
printNumericTable(qualityMet["explainedVariance"],"Explained Variance")
printNumericTable(qualityMet["explainedVarianceRatio"],"Explained Variance Ratio")
printNumericTable(qualityMet["noiseVariance"],"Noise Variance")
# Transform, using 2 principle componenests
transform_nT = pca.transform(pcaResult,X,nComponents=2)
printNumericTable(transform_nT, "transformed data",10)

# Serialize and save the pca result
pca.serialize(pcaResult, fileName = 'pcaResult.npy')
# Deserialize
de_pcaResult = pca.deserialize(fileName = "pcaResult.npy")