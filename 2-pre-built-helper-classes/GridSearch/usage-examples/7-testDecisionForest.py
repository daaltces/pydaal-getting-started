
import os
import sys
sys.path.append('..')
import daal.algorithms.decision_forest as df
from daal.algorithms.decision_forest.classification import prediction, training
from GridSearch import GridSearch
from daal.data_management import (
	FileDataSource, DataSourceIface, NumericTableIface, HomogenNumericTable,
	MergedNumericTable, data_feature_utils
)
# Input data set parameters
DATA_PREFIX = os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','data','batch')
# Input data set parameters
trainDatasetFileName = os.path.join(DATA_PREFIX, 'df_classification_train.csv')

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

#create a dictionary of hyperparameter values in a list

df_params = [{'nTrees': [10,100],
			'minObservationsInLeafNode':[8,10],
			'featuresPerNode':[0,3],
			'maxTreeDepth':[0],
			'minObservationsInLeafNode':[1,3],
			'impurityThreshold':[0,0.1],
			'resultsToCompute':[0],
			'varImportance':['MDI','MDA_Raw']}]			
			
#default keyword arguments
'''
GridSearch(<args>, tuned_parameters = None, score=None,
			best_score_criteria='high',
			create_best_training_model = False,
			save_model=False,nClasses=None )			
'''	
#create a dictionary of hyperparameter values in a list
	
clf = GridSearch(df,training,prediction, 
				tuned_parameters = df_params,score=None,
				best_score_criteria='high',
				create_best_training_model = True,
				save_model=True,nClasses=nClasses)		
result = clf.train(trainData,trainGroundTruth)
#Train on all combinations of hyperparameters
result = clf.train(trainData,trainGroundTruth)
#view all the parameters and scores from the best to worst order
result.viewAllResults()
#view the best parameters with score
print(result.bestResult())
