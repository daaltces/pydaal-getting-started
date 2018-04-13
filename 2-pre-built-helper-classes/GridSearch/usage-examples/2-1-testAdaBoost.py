import sys
sys.path.append('..')
import os
import daal.algorithms.adaboost as adaB
from daal.algorithms.adaboost import prediction, training
from GridSearch import GridSearch
from daal.data_management import (
	FileDataSource, DataSourceIface, HomogenNumericTable, MergedNumericTable, NumericTableIface
)


DATA_PREFIX = os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','data','batch')
trainDatasetFileName = os.path.join(DATA_PREFIX, 'adaboost_train.csv')

nFeatures = 20

# Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
trainDataSource = FileDataSource(
	trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
	DataSourceIface.doDictionaryFromContext
)

# Create Numeric Tables for training data and labels
trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
mergedData = MergedNumericTable(trainData, trainGroundTruth)

# Retrieve the data from the input file
trainDataSource.loadDataBlock(mergedData)

#default keyword arguments
'''
GridSearch(<args>, tuned_parameters = None, score=None,
			best_score_criteria='high',
			create_best_training_model = False,
			save_model=False,nClasses=None )			
'''	
#create a dictionary of hyperparameter values in a list
adaB_params = [{'accuracyThreshold': [0.99,0.1],
				'maxIterations' :[1,5]}]
#Create GridSearch object						 
clf = GridSearch(adaB,training,prediction, 
				tuned_parameters = adaB_params,score=None,
				best_score_criteria='high',
				create_best_training_model=True,
				save_model=True,nClasses=5)		
#Train on all combinations of hyperparameters
result = clf.train(trainData,trainGroundTruth)
#view all the parameters and scores in best to worst order
result.viewAllResults()
#view the best parameters with score
print(result.bestResult())



