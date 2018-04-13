import sys
sys.path.append('..')
import os
from daal.algorithms.brownboost import prediction, training
import daal.algorithms.brownboost as brownB
from GridSearch import GridSearch
from daal.data_management import (
	FileDataSource, DataSourceIface, HomogenNumericTable, MergedNumericTable, NumericTableIface
)



DATA_PREFIX = os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','data','batch')

trainDatasetFileName = os.path.join(DATA_PREFIX,'brownboost_train.csv')

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
brownB_params = [{'accuracyThreshold': [0.01,0.1,0.001],
				'maxIterations' :[1,20],
				'newtonRaphsonAccuracyThreshold':[1.0e-3,1.0e-2],
				'newtonRaphsonMaxIterations':[100,10],
				'degenerateCasesThreshold':[1.0e-2,1.0e-1]}]
#Create GridSearch object						 
clf = GridSearch(brownB,training,prediction, 
			tuned_parameters = brownB_params,score=None,
			best_score_criteria='high',
			create_best_training_model=True,
			save_model=True,nClasses=5)		
#Train on all combinations of hyperparameters
result = clf.train(trainData,trainGroundTruth)
#view all the parameters and scores in best to worst order
result.viewAllResults()
#view the best parameters with score
print(result.bestResult())




