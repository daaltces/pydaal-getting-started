import sys
sys.path.append('..')
import os
from GridSearch import GridSearch
from  daal.algorithms.multinomial_naive_bayes import training, prediction
import  daal.algorithms.multinomial_naive_bayes as mn

from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)


# Input data set parameters
DATA_PREFIX = os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','data','batch')
trainDatasetFileName = os.path.join(DATA_PREFIX, 'naivebayes_train_dense.csv')

nFeatures = 20
nClasses = 20

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

#create a dictionary of hyperparameter values in a list
mn_params = [{'priorClassEstimates': [HomogenNumericTable(20*[[0.05]])],
				'alpha':[HomogenNumericTable([20*[1]])]}]
				
#default keyword arguments
'''
GridSearch(<args>, tuned_parameters = None, score=None,
			best_score_criteria='high',
			create_best_training_model = False,
			save_model=False,nClasses=None )			
'''				
#Create GridSearch object	
#Flags create_best_training_model= True, save_model= True are unavailable for Naive Bayes as of yet 			
clf = GridSearch(mn,training,prediction, 
				tuned_parameters = mn_params,score=None,
				best_score_criteria='high',
				create_best_training_model=False,
				save_model=False,nClasses=20)		
#Train on all combinations of hyperparameters
result = clf.train(trainData,trainGroundTruth)
#view all the parameters and scores in best to worst order
result.viewAllResults()
#view the best parameters with score
print(result.bestResult())



