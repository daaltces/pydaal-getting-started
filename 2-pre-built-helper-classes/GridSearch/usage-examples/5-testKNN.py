
import os
import sys
sys.path.append('..')
import daal.algorithms.kdtree_knn_classification as knn
from daal.algorithms.kdtree_knn_classification import training, prediction
from GridSearch import GridSearch
from daal.data_management import (
	FileDataSource, DataSourceIface, NumericTableIface, HomogenNumericTable, MergedNumericTable
)

DATA_PREFIX = os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','data','batch')

# Input data set parameters
trainDatasetFileName = os.path.join(DATA_PREFIX,  'k_nearest_neighbors_train.csv')

nFeatures = 5

# Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
trainDataSource = FileDataSource(
	trainDatasetFileName,
	DataSourceIface.notAllocateNumericTable,
	DataSourceIface.doDictionaryFromContext
)

# Create Numeric Tables for training data and labels
trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
mergedData = MergedNumericTable(trainData, trainGroundTruth)

# Retrieve the data from the input file
trainDataSource.loadDataBlock(mergedData)

knn_params = [{'dataUseInModel': ['doNotUse','doUse']},{'k':[1,2,3]}]	
#Create GridSearch object						 
clf = GridSearch(knn,training,prediction, 
				tuned_parameters = knn_params,score=None,
				best_score_criteria='high',
				create_best_training_model=True,
				save_model=True)		
#Train on all combinations of hyperparameters
result = clf.train(trainData,trainGroundTruth)
#view all the parameters and scores in best to worst order
result.viewAllResults()
#view the best parameters with score
print(result.bestResult())
