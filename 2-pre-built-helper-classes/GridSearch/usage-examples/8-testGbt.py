
import os
import sys
sys.path.append('..')
import daal.algorithms.gbt as gbt
from daal.algorithms.gbt.classification import prediction, training
from GridSearch import GridSearch
from daal.data_management import (
	FileDataSource, DataSourceIface, NumericTableIface, HomogenNumericTable,
	MergedNumericTable, data_feature_utils
)

DATA_PREFIX = os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','data','batch')

# Input data set parameters
trainDatasetFileName = os.path.join(DATA_PREFIX,  'df_classification_train.csv')

nFeatures = 3
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

#  Get the dictionary and update it with additional information about data
dict = trainData.getDictionary()

#  Add a feature type to the dictionary
dict[0].featureType = data_feature_utils.DAAL_CONTINUOUS
dict[1].featureType = data_feature_utils.DAAL_CONTINUOUS
dict[2].featureType = data_feature_utils.DAAL_CATEGORICAL


#default keyword arguments
'''
GridSearch(<args>, tuned_parameters = None, score=None,
			best_score_criteria='high',
			create_best_training_model = False,
			save_model=False,nClasses=None )	
'''
#create a dictionary of hyperparameter values in a list
#Note that the list has GBT prediction parameters 
#alongwith training parmaeters, pediction parameters are also tuned, 
#For more details refer to the documentation
gbt_params = [{'maxIterations': [50,100],
			'maxTreeDepth':[6],
			'shrinkage':[0.3],
			'minSplitLoss':[0],
			'observationsPerTreeFraction':[1],
			'featuresPerNode':[0],
			'minObservationsInLeafNode':[5],
			'memorySavingMode':[False]},
			{'numIterations':[0, 30]}]


#Create GridSearch object						 
clf = GridSearch(gbt,training,prediction, 
				tuned_parameters = gbt_params,score=None,
				best_score_criteria='high',
				create_best_training_model=True,
				save_model=True,nClasses=nClasses)		
#Train on all combinations of hyperparameters
result = clf.train(trainData,trainGroundTruth)
#view all the parameters and scores in best to worst order
result.viewAllResults()
#view the best parameters with score
print(result.bestResult())









