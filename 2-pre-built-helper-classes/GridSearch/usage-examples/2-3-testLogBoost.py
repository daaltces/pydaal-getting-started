
import os
import sys
sys.path.append('..')
from daal.algorithms.logitboost import prediction, training
import daal.algorithms.logitboost as logB
from daal.algorithms import classifier
from daal.data_management import (
	FileDataSource, DataSourceIface, NumericTableIface, HomogenNumericTable, MergedNumericTable
)
from GridSearch import GridSearch

DATA_PREFIX = os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','data','batch')

# Input data set parameters
trainDatasetFileName = os.path.join(DATA_PREFIX, 'logitboost_train.csv')
nFeatures = 20
nClasses = 5

# LogitBoost algorithm parameters
maxIterations = 100       # Maximum number of terms in additive regression
accuracyThreshold = 0.01  # Training accuracy

# Model object for the LogitBoost algorithm
model = None
predictionResult = None
testGroundTruth = None


# Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
trainDataSource = FileDataSource(
	trainDatasetFileName,
	DataSourceIface.notAllocateNumericTable,
	DataSourceIface.doDictionaryFromContext
)

# Create Numeric Tables for training data and labels
trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
mergedData = MergedNumericTable(trainData, trainGroundTruth)

# Retrieve the data from the input file
trainDataSource.loadDataBlock(mergedData)
		
logB_params = [{'accuracyThreshold': [0.01,0.1,0.001],
				'maxIterations' :[2,20],
				'weightsDegenerateCasesThreshold':[1e-10,1e-8],
				'responsesDegenerateCasesThreshold':[1e-10]}]	

#default keyword arguments
'''
GridSearch(<args>, tuned_parameters = None, score=None,
			best_score_criteria='high',
			create_best_training_model = False,
			save_model=False,nClasses=None )
'''	
				
clf = GridSearch(logB,training,prediction, tuned_parameters = logB_params,score=None,best_score_criteria='high',nClasses=5)		
result = clf.train(trainData,trainGroundTruth)
result.viewAllResults()
print(result.bestResult())					 
