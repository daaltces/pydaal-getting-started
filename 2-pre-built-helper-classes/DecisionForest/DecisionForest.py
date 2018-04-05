import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','python','source'))
from  daal.algorithms import decision_forest, classifier
from daal.algorithms.decision_forest.classification import training as training_cl 
from daal.algorithms.decision_forest.classification import prediction as prediction_cl
from daal.algorithms.decision_forest.regression  import training, prediction
from daal.data_management import InputDataArchive, OutputDataArchive
from daal.data_management import Compressor_Zlib, Decompressor_Zlib, level9, DecompressionStream, CompressionStream, \
	HomogenNumericTable, readOnly
from daal.data_management import BlockDescriptor, readWrite

from utils import printNumericTable
import numpy as np
from numpy import float32, float64
import warnings

from daal.algorithms.multi_class_classifier import quality_metric_set as multiclass_quality
from daal.algorithms.classifier.quality_metric import multiclass_confusion_matrix 
from daal.algorithms.svm import quality_metric_set as twoclass_quality
from daal.algorithms.classifier.quality_metric import binary_confusion_matrix
from daal.data_management import BlockDescriptor_Float64, readOnly

from collections import namedtuple



# Two-class quality metrics type
TwoClassMetrics = namedtuple('TwoClassMetrics',
		['accuracy', 'precision', 'recall', 'fscore', 'specificity', 'auc'])

# Multi-class quality metrics type
MultiClassMetrics = namedtuple('MultiClassMetrics',
		['averageAccuracy', 'errorRate', 'microPrecision', 'microRecall',
		 'microFscore', 'macroPrecision', 'macroRecall', 'macroFscore'])


class ClassifierQualityMetrics:


	def __init__(self, truth, predictions, nclasses = 2):
		"""Initialize class parameters

		Args:
		   truth: ground truth
		   predictions: predicted labels
		   nclasses: number of classes
		"""

		self._truth = truth
		self._predictions = predictions
		if nclasses == 2:
			self._computeTwoclassQualityMetrics()
		elif nclasses > 2:
			self._computeMulticlassQualityMetrics(nclasses)
		else:
			raise ValueError('nclasses must be at least 2')


	def get(self, metric):
		"""Get a metric from the quality metrics collection

		Args:
		   metric: name of the metric to return

		Returns:
		   A numeric value for the given metric
		"""
		if metric is not 'confusionMatrix':
			return getattr(self._metrics, metric)
		else:
			return self._confMat
			
	def getAllMetrics(self):
		return self._metrics

	def _computeTwoclassQualityMetrics(self):
		# Alg object for quality metrics computation
		quality_alg = twoclass_quality.Batch()
		# Get access to the input parameter
		input = quality_alg.getInputDataCollection().getInput(
				twoclass_quality.confusionMatrix)
		# Pass ground truth and predictions as input
		input.set(binary_confusion_matrix.groundTruthLabels, self._truth)
		input.set(binary_confusion_matrix.predictedLabels, self._predictions)
		# Compute confusion matrix
		confusion = quality_alg.compute().getResult(twoclass_quality.confusionMatrix)
		#confusion matrix
		self._confMat = confusion.get(binary_confusion_matrix.confusionMatrix)
		# Retrieve quality metrics from the confusion matrix		
		metrics = confusion.get(binary_confusion_matrix.binaryMetrics)
		# Convert the metrics into a Python namedtuple and return it
		block = BlockDescriptor_Float64()
		metrics.getBlockOfRows(0, 1, readOnly, block)
		x = block.getArray().flatten()
		self._metrics = TwoClassMetrics(*x)
		print(self._metrics)
		metrics.releaseBlockOfRows(block)



	def _computeMulticlassQualityMetrics(self, nclasses):
		# Alg object for quality metrics computation
		quality_alg = multiclass_quality.Batch(nclasses)
		# Get access to the input parameter
		input = quality_alg.getInputDataCollection().getInput(
				multiclass_quality.confusionMatrix)
		# Pass ground truth and predictions as input
		input.set(multiclass_confusion_matrix.groundTruthLabels, self._truth)
		input.set(multiclass_confusion_matrix.predictedLabels, self._predictions)
		# Compute confusion matrix
		confusion = quality_alg.compute().getResult(multiclass_quality.confusionMatrix)
		#confusion Matrix
		self._confMat = confusion.get(multiclass_confusion_matrix.confusionMatrix)
		# Retrieve quality metrics from the confusion matrix
		metrics = confusion.get(multiclass_confusion_matrix.multiClassMetrics)
		# Convert the metrics into a Python namedtuple and return it
		block = BlockDescriptor_Float64()
		metrics.getBlockOfRows(0, 1, readOnly, block)
		x = block.getArray().flatten()
		self._metrics = MultiClassMetrics(*x)
		metrics.releaseBlockOfRows(block)
		
class Classification:
	'''
	Constructor to set Decision forest paramters classification training parameters
	'''

	def __init__(self, nClasses, seed=777, nTrees = 100, observationsPerTreeFraction = 1,featuresPerNode=0,maxTreeDepth=0,
				 minObservationsInLeafNodes=1,impurityThreshold=0,varImportance=None,resultsToCompute=0):
		'''
		seed: default: 777
				The seed for random number generator, which is used to choose the bootstrap set, split features in every split node in a tree, 
				and generate permutation required in computations of MDA variable importance.
		nTrees: default: 100
			The number of trees in the forest.
		observationsPerTreeFraction: default: 1
			Fraction of the training set S used to form the bootstrap set for a single tree training, 0 < observationsPerTreeFraction <= 1. 
			The observations are sampled randomly with replacement.
		featuresPerNode:default: 0
			The number of features tried as possible splits per node. 
			If the parameter is set to 0, the library uses the square root of the number of features for classification and (the number of features)/3 for regression.
		maxTreeDepth : default: 0
			Maximal tree depth. Default is 0 (unlimited).
		minObservationsInLeafNodes: default: 1
			Minimum number of observations in the leaf node.
		impurityThreshold: default: 0
			The threshold value used as stopping criteria: 
			if the impurity value in the node is smaller than the threshold, the node is not split anymore.
		varImportance: default: none
			The variable importance computation mode.
			Possible values:
			•	None or 0– variable importance is not calculated
			•	‘MDI’ or 1- also known as the Gini importance or Mean Decrease Gini
			•	‘MDA_Raw’ or 2 - Mean Decrease of Accuracy (permutation importance)
			•	‘MDA_Scaled’ or 3 - the MDA_Raw value scaled by its standard deviation
		resultsToCompute: default: 0
			Possible values
			•	None or 0- no OOB error calculated
			•	1 -  OOB error calculated
			•	2 -  OOB error per observation calculated
			•	3 -  OOB error and OOB error per observation calculated		
		'''
				 
		self.classes = nClasses
		self.seed = seed
		self.nTrees = nTrees
		self.observationsPerTreeFraction = observationsPerTreeFraction
		self.featuresPerNode = featuresPerNode
		self.maxTreeDepth = maxTreeDepth
		self.minObservationsInLeafNodes = minObservationsInLeafNodes
		self.impurityThreshold = impurityThreshold
		self.varImportance = varImportance
		self.resultsToCompute=resultsToCompute
		
	'''
	Arguments: train data feature values(type nT), train data target values(type nT)
	Returns training results object. Refer the developers guide to know how to get OOB error and other information
	'''
	def training(self, trainData, trainDependentVariables):
		trainingBatch = training_cl.Batch (self.classes)
		trainingBatch.parameter.seed = self.seed
		trainingBatch.parameter.nTrees = self.nTrees
		trainingBatch.parameter.observationsPerTreeFraction = self.observationsPerTreeFraction
		trainingBatch.parameter.featuresPerNode = self.featuresPerNode
		trainingBatch.parameter.maxTreeDepth = self.maxTreeDepth
		trainingBatch.parameter.minObservationsInLeafNodes = self.minObservationsInLeafNodes
		trainingBatch.parameter.impurityThreshold = self.impurityThreshold
		trainingBatch.parameter.resultsToCompute = self.resultsToCompute
		
		#trainingBatch.parameter.resultsToCompute = self.resultsToCompute
		if self.varImportance == None or self.varImportance == 0:
			trainingBatch.parameter.varImportance = decision_forest.training.none
		elif self.varImportance == 'MDI'or self.varImportance==1:
			trainingBatch.parameter.varImportance = decision_forest.training.MDI
		elif self.varImportance == 'MDA_Raw' or self.varImportance==2:
			trainingBatch.parameter.varImportance = decision_forest.training.MDA_Raw
		elif self.varImportance == 'MDA_Scaled' or self.varImportance==3:
			trainingBatch.parameter.varImportance = decision_forest.training.MDA_Scaled
		else:
			warnings.warn ('Incorrect varImportance argument passed. Set to default argument "MDI"')
			trainingBatch.parameter.varImportance = decision_forest.training.MDI
		trainingBatch.input.set (classifier.training.data, trainData)
		trainingBatch.input.set (classifier.training.labels, trainDependentVariables)
		trainingResult = trainingBatch.compute()
		if self.varImportance is not None and self.varImportance != 'none':
			printNumericTable (trainingResult.getTable (training_cl.variableImportance), "Variable importance results: ")
		if self.resultsToCompute ==1:			
			printNumericTable (trainingResult.getTable (training_cl.outOfBagError), "OOB error: ")
		elif self.resultsToCompute ==2:			
			printNumericTable (trainingResult.getTable (training_cl.outOfBagErrorPerObservation), "OOB error per observation: ",10)		
			warnings.warn ("To get all OOBErrorPerObervations, use the method trainingResult.getTable (training_cl.outOfBagErrorPerObservation")			
		elif self.resultsToCompute ==3:
			printNumericTable (trainingResult.getTable (training_cl.outOfBagError), "OOB error: ")		
			printNumericTable (trainingResult.getTable (training_cl.outOfBagErrorPerObservation), "OOB error per observation: ",10)			
			warnings.warn ("To get all OOBErrorPerObervations, use the method trainingResult.getTable (training_cl.outOfBagErrorPerObservation")
		return trainingResult
	'''
	Arguments: training result object, test data feature values(type nT)
	Returns predicted values(type nT)
	'''

	def predict(self, trainingResult, testData):  # give other parameters\		
		algorithm = prediction_cl.Batch (self.classes)
		algorithm.input.setTable (classifier.prediction.data, testData)
		algorithm.input.setModel (classifier.prediction.model, trainingResult.get (classifier.training.model))
		algorithm.compute ()
		predictionResult = algorithm.getResult ()
		predictedResponses = predictionResult.get(classifier.prediction.prediction)
		return predictedResponses

	'''
	Arguments: deserialized numpy array
	Returns decompressed numpy array
	'''

	def compress(self, arrayData):
		compressor = Compressor_Zlib ()
		compressor.parameter.gzHeader = True
		compressor.parameter.level = level9
		comprStream = CompressionStream (compressor)
		comprStream.push_back (arrayData)
		compressedData = np.empty (comprStream.getCompressedDataSize (), dtype=np.uint8)
		comprStream.copyCompressedArray (compressedData)
		return compressedData

	'''
	Arguments: serialized numpy array
	Returns Compressed numpy array
	'''

	def decompress(self, arrayData):
		decompressor = Decompressor_Zlib ()
		decompressor.parameter.gzHeader = True
		# Create a stream for decompression
		deComprStream = DecompressionStream (decompressor)
		# Write the compressed data to the decompression stream and decompress it
		deComprStream.push_back (arrayData)
		# Allocate memory to store the decompressed data
		bufferArray = np.empty (deComprStream.getDecompressedDataSize (), dtype=np.uint8)
		# Store the decompressed data
		deComprStream.copyDecompressedArray (bufferArray)
		return bufferArray

	'''
	Method 1:
		   Arguments: data(type nT/model)
		   Returns serialized numpy array
	Method 2:
		   Arguments: data(type nT/model), fileName(.npy file to save serialized array to disk)
		   Saves serialized numpy array as "fileName" argument
	Method 3:
		   Arguments: data(type nT/model), useCompression = True
		   Returns compressed numpy array
	Method 4:
		   Arguments: data(type nT/model), fileName(.npy file to save serialized array to disk), useCompression = True
		   Saves compressed numpy array as "fileName" argument
	'''

	def serialize(self, data, fileName=None, useCompression=False):
		buffArrObjName = (str (type (data)).split ()[1].split ('>')[0] + "()").replace ("'", '')
		dataArch = InputDataArchive ()
		data.serialize (dataArch)
		length = dataArch.getSizeOfArchive ()
		bufferArray = np.zeros (length, dtype=np.ubyte)
		dataArch.copyArchiveToArray (bufferArray)
		if useCompression == True:
			if fileName != None:
				if len (fileName.rsplit (".", 1)) == 2:
					fileName = fileName.rsplit (".", 1)[0]
				compressedData = Classification.compress (self, bufferArray)
				np.save (fileName, compressedData)
			else:
				comBufferArray = Classification.compress (self, bufferArray)
				serialObjectDict = {"Array Object": comBufferArray,
									"Object Information": buffArrObjName}
				return serialObjectDict
		else:
			if fileName != None:
				if len (fileName.rsplit (".", 1)) == 2:
					fileName = fileName.rsplit (".", 1)[0]
				np.save (fileName, bufferArray)
			else:
				serialObjectDict = {"Array Object": bufferArray,
									"Object Information": buffArrObjName}
				return serialObjectDict
		infoFile = open (fileName + ".txt", "w")
		infoFile.write (buffArrObjName)
		infoFile.close ()

	'''
	Arguments: can be serialized/ compressed numpy array or serialized/ compressed .npy file saved to disk
	Returns deserialized/ decompressed numeric table/model    
	'''

	def deserialize(self, serialObjectDict=None, fileName=None, useCompression=False):
		import daal
		if fileName != None and serialObjectDict == None:
			bufferArray = np.load (fileName)
			buffArrObjName = open (fileName.rsplit (".", 1)[0] + ".txt", "r").read ()
		elif fileName == None and any (serialObjectDict):
			bufferArray = serialObjectDict["Array Object"]
			buffArrObjName = serialObjectDict["Object Information"]
		else:
			warnings.warn ('Expecting "bufferArray" or "fileName" argument, NOT both')
			raise SystemExit
		if useCompression == True:
			bufferArray = Classification.decompress (self, bufferArray)
		dataArch = OutputDataArchive (bufferArray)
		try:
			deSerialObj = eval (buffArrObjName)
		except AttributeError:
			deSerialObj = HomogenNumericTable ()
		deSerialObj.deserialize (dataArch)
		return deSerialObj

	'''
	Arguments: prediction values(type nT), test data actual target values(type nT)
	Returns qualityMetrics object
	'''	
	def qualityMetrics(self, predictResults, testGroundTruth):
		self._qualityMetricSetResult = ClassifierQualityMetrics(testGroundTruth, predictResults, self.classes)
		return self._qualityMetricSetResult			
		
	'''
	Arguments: training result object, test data feature values of type nT, test data actual target values(type nT)
	Returns predicted values(type nT), quality metrics object for binary classifier 
	'''
	def predictWithQualityMetrics(self, trainingResult, testData,testGroundTruth):

		# Retrieve predicted labels

		predictResults = self.predict (trainingResult, testData)
		self._qualityMetricSetResult=self.qualityMetrics(predictResults, testGroundTruth)
		return predictResults, self._qualityMetricSetResult
	'''
	Arguments: quality metrics object for binary classifier 
	Prints Accuracy, Precision, Recall, F1-score, Specificity, AUC
	'''
	def printAllQualityMetrics(self,qualityMetricSetResult):

		# Print the quality metrics
		if self.classes ==2:

			printNumericTable(qualityMetricSetResult.get('confusionMatrix'), "Confusion matrix:")

			print("Accuracy:      {0:.3f}".format(qualityMetricSetResult.get('accuracy')))
			print("Precision:     {0:.3f}".format(qualityMetricSetResult.get('precision')))
			print("Recall:        {0:.3f}".format(qualityMetricSetResult.get('recall')))
			print("F1-score:      {0:.3f}".format(qualityMetricSetResult.get('fscore')))
			print("Specificity:   {0:.3f}".format(qualityMetricSetResult.get('specificity')))
			print("AUC:           {0:.3f}".format(qualityMetricSetResult.get('auc')))
		
		else:
		
			printNumericTable(qualityMetricSetResult.get('confusionMatrix'), "Confusion matrix:")

			print ("Average accuracy: {0:.3f}".format (qualityMetricSetResult.get('averageAccuracy')))
			print ("Error rate:       {0:.3f}".format (qualityMetricSetResult.get('errorRate')))
			print ("Micro precision:  {0:.3f}".format (qualityMetricSetResult.get('microPrecision')))
			print ("Micro recall:     {0:.3f}".format (qualityMetricSetResult.get('microRecall')))
			print ("Micro F-score:    {0:.3f}".format (qualityMetricSetResult.get('microFscore')))
			print ("Macro precision:  {0:.3f}".format (qualityMetricSetResult.get('macroPrecision')))
			print ("Macro recall:     {0:.3f}".format (qualityMetricSetResult.get('macroRecall')))
			print ("Macro F-score:    {0:.3f}".format (qualityMetricSetResult.get('macroFscore')))		
		

class Regression:
	'''
	Constructor to set Decision forest paramters regression training parameters
	'''
	def __init__(self, seed=777, nTrees = 100, observationsPerTreeFraction = 1,featuresPerNode=0,maxTreeDepth=0,
				 minObservationsInLeafNodes=5,impurityThreshold=0,varImportance=None,resultsToCompute=0):
		'''
		seed: default: 777
				The seed for random number generator, which is used to choose the bootstrap set, split features in every split node in a tree, 
				and generate permutation required in computations of MDA variable importance.
		nTrees: default: 100
			The number of trees in the forest.
		observationsPerTreeFraction: default: 1
			Fraction of the training set S used to form the bootstrap set for a single tree training, 0 < observationsPerTreeFraction <= 1. 
			The observations are sampled randomly with replacement.
		featuresPerNode:default: 0
			The number of features tried as possible splits per node. 
			If the parameter is set to 0, the library uses the square root of the number of features for classification and (the number of features)/3 for regression.
		maxTreeDepth : default: 0
			Maximal tree depth. Default is 0 (unlimited).
		minObservationsInLeafNodes: default: 5
			Minimum number of observations in the leaf node.
		impurityThreshold: default: 0
			The threshold value used as stopping criteria: 
			if the impurity value in the node is smaller than the threshold, the node is not split anymore.
		varImportance: default: none
			The variable importance computation mode.
			Possible values:
			•	None or 0 – variable importance is not calculated
			•	‘MDI’ or 1- also known as the Gini importance or Mean Decrease Gini
			•	‘MDA_Raw’ or 2 - Mean Decrease of Accuracy (permutation importance)
			•	‘MDA_Scaled’ or 3 - the MDA_Raw value scaled by its standard deviation
		resultsToCompute: default: 0
			Possible values
			•	None or 0- no OOB error calculated
			•	1 -  OOB error calculated
			•	2 -  OOB error per observation calculated
			•	3 -  OOB error and OOB error per observation calculated		
		'''		 
		self.seed = seed
		self.nTrees = nTrees
		self.observationsPerTreeFraction = observationsPerTreeFraction
		self.featuresPerNode = featuresPerNode
		self.maxTreeDepth = maxTreeDepth
		self.minObservationsInLeafNodes = minObservationsInLeafNodes
		self.impurityThreshold = impurityThreshold
		self.varImportance = varImportance
		self.resultsToCompute=resultsToCompute
	'''
	Arguments: train data feature values(type nT), train data target values(type nT)
	Returns training results object. Refer the developers guide to know how to get OOB error and other information
	'''
	def training(self, trainData, trainDependentVariables):
		trainingBatch = training.Batch ()
		trainingBatch.parameter.seed = self.seed
		trainingBatch.parameter.nTrees = self.nTrees
		trainingBatch.parameter.observationsPerTreeFraction = self.observationsPerTreeFraction
		trainingBatch.parameter.featuresPerNode = self.featuresPerNode
		trainingBatch.parameter.maxTreeDepth = self.maxTreeDepth
		trainingBatch.parameter.minObservationsInLeafNodes = self.minObservationsInLeafNodes
		trainingBatch.parameter.impurityThreshold = self.impurityThreshold
		trainingBatch.parameter.resultsToCompute = self.resultsToCompute
		
		if self.varImportance == None or self.varImportance == 0:
			trainingBatch.parameter.varImportance = decision_forest.training.none
		elif self.varImportance == 'MDI'or self.varImportance==1:
			trainingBatch.parameter.varImportance = decision_forest.training.MDI
		elif self.varImportance == 'MDA_Raw' or self.varImportance==2:
			trainingBatch.parameter.varImportance = decision_forest.training.MDA_Raw
		elif self.varImportance == 'MDA_Scaled' or self.varImportance==3:
			trainingBatch.parameter.varImportance = decision_forest.training.MDA_Scaled
		else:
			warnings.warn ('Incorrect varImportance argument passed. Set to default argument "MDI"')
			trainingBatch.parameter.varImportance = decision_forest.training.MDI
		trainingBatch.input.set (training.data, trainData)
		trainingBatch.input.set (training.dependentVariable, trainDependentVariables)
		trainingResult = trainingBatch.compute()
		if self.varImportance is not None and self.varImportance != 'none':
			printNumericTable (trainingResult.getTable (training.variableImportance), "Variable importance results: ")
		if self.resultsToCompute ==1:			
			printNumericTable (trainingResult.getTable (training.outOfBagError), "OOB error: ")
		elif self.resultsToCompute ==2:			
			printNumericTable (trainingResult.getTable (training.outOfBagErrorPerObservation), "OOB error per observation: ",10)		
			warnings.warn ("To get all OOBErrorPerObervations, use the method trainingResult.getTable (training.outOfBagErrorPerObservation")			
		elif self.resultsToCompute ==3:
			printNumericTable (trainingResult.getTable (training.outOfBagError), "OOB error: ")		
			printNumericTable (trainingResult.getTable (training.outOfBagErrorPerObservation), "OOB error per observation: ",10)			
			warnings.warn ("To get all OOBErrorPerObervations, use the method trainingResult.getTable (training.outOfBagErrorPerObservation")
		return trainingResult
	'''
	Arguments: training result object, test data feature values(type nT)
	Returns predicted values(type nT)
	'''	

	def predict(self, trainingResult, testData):  # give other parameters
		from daal.algorithms.decision_forest.regression import training, prediction
		algorithm = prediction.Batch ()
		algorithm.input.setTable (prediction.data, testData)
		algorithm.input.setModel (prediction.model, trainingResult.get (training.model))
		algorithm.compute ()
		predictionResult = algorithm.getResult ()
		predictedResponses = predictionResult.get (prediction.prediction)
		return predictedResponses

	'''
	Arguments: deserialized numpy array
	Returns decompressed numpy array
	'''

	def compress(self, arrayData):
		compressor = Compressor_Zlib ()
		compressor.parameter.gzHeader = True
		compressor.parameter.level = level9
		comprStream = CompressionStream (compressor)
		comprStream.push_back (arrayData)
		compressedData = np.empty (comprStream.getCompressedDataSize (), dtype=np.uint8)
		comprStream.copyCompressedArray (compressedData)
		return compressedData

	'''
	Arguments: serialized numpy array
	Returns Compressed numpy array
	'''

	def decompress(self, arrayData):
		decompressor = Decompressor_Zlib ()
		decompressor.parameter.gzHeader = True
		# Create a stream for decompression
		deComprStream = DecompressionStream (decompressor)
		# Write the compressed data to the decompression stream and decompress it
		deComprStream.push_back (arrayData)
		# Allocate memory to store the decompressed data
		bufferArray = np.empty (deComprStream.getDecompressedDataSize (), dtype=np.uint8)
		# Store the decompressed data
		deComprStream.copyDecompressedArray (bufferArray)
		return bufferArray

	'''
	Method 1:
		   Arguments: data(type nT/model)
		   Returns serialized numpy array
	Method 2:
		   Arguments: data(type nT/model), fileName(.npy file to save serialized array to disk)
		   Saves serialized numpy array as "fileName" argument
	Method 3:
		   Arguments: data(type nT/model), useCompression = True
		   Returns compressed numpy array
	Method 4:
		   Arguments: data(type nT/model), fileName(.npy file to save serialized array to disk), useCompression = True
		   Saves compressed numpy array as "fileName" argument
	'''

	def serialize(self, data, fileName=None, useCompression=False):
		buffArrObjName = (str (type (data)).split ()[1].split ('>')[0] + "()").replace ("'", '')
		dataArch = InputDataArchive ()
		data.serialize (dataArch)
		length = dataArch.getSizeOfArchive ()
		bufferArray = np.zeros (length, dtype=np.ubyte)
		dataArch.copyArchiveToArray (bufferArray)
		if useCompression == True:
			if fileName != None:
				if len (fileName.rsplit (".", 1)) == 2:
					fileName = fileName.rsplit (".", 1)[0]
				compressedData = Regression.compress (self, bufferArray)
				np.save (fileName, compressedData)
			else:
				comBufferArray = Regression.compress (self, bufferArray)
				serialObjectDict = {"Array Object": comBufferArray,
									"Object Information": buffArrObjName}
				return serialObjectDict
		else:
			if fileName != None:
				if len (fileName.rsplit (".", 1)) == 2:
					fileName = fileName.rsplit (".", 1)[0]
				np.save (fileName, bufferArray)
			else:
				serialObjectDict = {"Array Object": bufferArray,
									"Object Information": buffArrObjName}
				return serialObjectDict
		infoFile = open (fileName + ".txt", "w")
		infoFile.write (buffArrObjName)
		infoFile.close ()

	'''
	Arguments: can be serialized/ compressed numpy array or serialized/ compressed .npy file saved to disk
	Returns deserialized/ decompressed numeric table/model    
	'''

	def deserialize(self, serialObjectDict=None, fileName=None, useCompression=False):
		import daal
		if fileName != None and serialObjectDict == None:
			bufferArray = np.load (fileName)
			buffArrObjName = open (fileName.rsplit (".", 1)[0] + ".txt", "r").read ()
		elif fileName == None and any (serialObjectDict):
			bufferArray = serialObjectDict["Array Object"]
			buffArrObjName = serialObjectDict["Object Information"]
		else:
			warnings.warn ('Expecting "bufferArray" or "fileName" argument, NOT both')
			raise SystemExit
		if useCompression == True:
			bufferArray = Regression.decompress (self, bufferArray)
		dataArch = OutputDataArchive (bufferArray)
		try:
			deSerialObj = eval (buffArrObjName)
		except AttributeError:
			deSerialObj = HomogenNumericTable ()
		deSerialObj.deserialize (dataArch)
		return deSerialObj

	
	
	
	