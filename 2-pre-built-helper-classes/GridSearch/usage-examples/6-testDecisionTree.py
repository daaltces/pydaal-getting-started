
import os
import sys
sys.path.append('..')
import daal.algorithms.decision_tree.classification as dt
from daal.algorithms.decision_tree.classification import prediction, training
from GridSearch import GridSearch
from daal.data_management import (
	FileDataSource, DataSourceIface, NumericTableIface, HomogenNumericTable, MergedNumericTable
)
DATA_PREFIX = os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','data','batch')

# Input data set parameters
trainDatasetFileName = os.path.join(DATA_PREFIX,'decision_tree_train.csv')
pruneDatasetFileName = os.path.join(DATA_PREFIX, 'decision_tree_prune.csv')

nFeatures = 5
nClasses = 5



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

# Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
pruneDataSource = FileDataSource(
	pruneDatasetFileName,
	DataSourceIface.notAllocateNumericTable,
	DataSourceIface.doDictionaryFromContext
)

# Create Numeric Tables for pruning data and labels
pruneData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
pruneGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
pruneMergedData = MergedNumericTable(pruneData, pruneGroundTruth)

# Retrieve the data from the input file
pruneDataSource.loadDataBlock(pruneMergedData)
#create a dictionary of hyperparameter values in a list
dt_params = [{'splitCriterion': ['infoGain'],
			'maxTreeDepth':[0,10],
			'minObservationsInLeafNodes':[1,5]}]	
#Create GridSearch object				
clf = GridSearch(dt,training,prediction, 
				tuned_parameters = dt_params,score=None,
				best_score_criteria='high',
				create_best_training_model=True,
				save_model=True,nClasses=nClasses)				
#Train on all combinations of hyperparameters
result = clf.train(trainData,trainGroundTruth,pruneData,pruneGroundTruth)
#view all the parameters and scores in best to worst order
result.viewAllResults()
#view the best parameters with score
print(result.bestResult())
