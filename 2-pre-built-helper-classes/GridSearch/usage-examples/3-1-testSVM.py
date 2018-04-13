
import os
import sys
sys.path.append('..')
from GridSearch import GridSearch
from  daal.algorithms.svm import training, prediction
import   daal.algorithms.svm as svm
from daal.data_management import (
	FileDataSource, DataSourceIface, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

DATA_PREFIX = os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','data','batch')

trainDatasetFileName = os.path.join(DATA_PREFIX,'svm_two_class_train_dense.csv')

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
svm_params = [{'C':[0.5,1],
				'accuracyThreshold':[0.01,0.001],
				'cacheSize':[600000000],
				'tau':[1.0e-6,1.0e-5],
				'maxIterations':[100,10],
				'doShrinking':[True, False]}]	
#Create GridSearch object						 
clf = GridSearch(svm,training,prediction, 
				tuned_parameters = svm_params,score=None,
				best_score_criteria='high',
				create_best_training_model=True,
				save_model=True,nClasses=None)		
#Train on all combinations of hyperparameters
result = clf.train(trainData,trainGroundTruth)
#view all the parameters and scores in best to worst order
result.viewAllResults()
#view the best parameters with score
print(result.bestResult())




