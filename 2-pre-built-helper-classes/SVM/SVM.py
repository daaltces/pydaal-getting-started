import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','python','source'))
from utils import printNumericTable
from numpy import float32, float64
import numpy as np
import warnings
from daal.algorithms.svm import training, prediction
from daal.algorithms import kernel_function, classifier, multi_class_classifier
from daal.data_management import (BlockDescriptor_Float64, BlockDescriptor, readOnly, readWrite, InputDataArchive, OutputDataArchive, Compressor_Zlib, Decompressor_Zlib,
								  level9, DecompressionStream, CompressionStream, HomogenNumericTable)
from daal.algorithms.multi_class_classifier import quality_metric_set as multiclass_quality
from daal.algorithms.classifier.quality_metric import multiclass_confusion_matrix 
from daal.algorithms.svm import quality_metric_set as twoclass_quality
from daal.algorithms.classifier.quality_metric import binary_confusion_matrix

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
		

class BinarySVM:

	'''
	Constructor to set SVM training parameters
	parameters:
	
	'''
	def __init__(self, method="boser", C = 1, tolerence = 0.001, tau = 0.000001, maxIterations = 1000000, cacheSize = 8000000, doShrinking = True, kernel = 'linear',
				 sigma = 0,k = 1, b = 0,dtype = float64):
		'''
		method: 'boser', default: 'boser'
			computation method
		C: deafult: 1
			Upper bound in conditions of the quadratic optimization problem.
		tolerance: default: '0.001'
			Training accuracy/ stopping criteria
		tau: default: 0.000001
			Tau parameter of the WSS scheme.
		maxiterations: default: 1000000
			Maximal number of iterations for the algorithm.
		cacheSize: default: 8000000
			cachesize for storing values of kernal matrix.
		doShringing: True/false, default: True
			flag to set shrinking optimization techinique
		kernel: 'linear'/'rbf', default: 'linear
		k: default: 1
			coefficient value of k when kernal function is 'linear'
		b: 	default: 0
			coeffiecent value of b of linear function
		dtype: intc, float32, float63, intc
		'''
		self.method = method
		# Print error message here"
		self.dtype = dtype
		self.C = C
		self.tolerence = tolerence
		self.tau = tau
		self.maxIterations = maxIterations
		self.cachesize = cacheSize
		self.doShrinking = doShrinking
		self.kernel = kernel
		if self.kernel == "rbf":
			self.sigma = sigma
		elif self.kernel == "linear":
			self.k = k
			self.b = b
	'''
	Arguments: train data feature values(type nT), train data target values(type nT)
	Returns training results object
	'''
	def training(self, trainData, trainDependentVariables):

		#Set algorithms parameters
		if self.method == 'boser':
			from  daal.algorithms.svm.training import boser
			algorithm = training.Batch (method=boser, fptype=self.dtype)   
		if self.kernel == 'linear':
			algorithm.parameter.kernel = kernel_function.linear.Batch (method=boser, fptype=self.dtype)
			algorithm.parameter.k = self.k
			algorithm.parameter.b = self.b
		elif self.kernel == 'rbf':
			algorithm.parameter.kernel =  kernel_function.rbf.Batch(method=boser, fptype=self.dtype)
			algorithm.parameter.sigma = self.sigma

		algorithm.parameter.cacheSize = self.cachesize
		algorithm.parameter.C = self.C
		algorithm.parameter.accuracyThreshold = self.tolerence
		algorithm.parameter.tau = self.tau
		algorithm.parameter.maxIterations = self.maxIterations
		algorithm.parameter.doShrinking = self.doShrinking

		algorithm.input.set (classifier.training.data, trainData)
		algorithm.input.set (classifier.training.labels, trainDependentVariables)
		trainingResult = algorithm.compute ()
		return trainingResult
	'''
	Arguments: training result object, test data feature values(type nT)
	Returns predicted values of type nT
	'''
	def predict(self, trainingResult, testData): 

		if self.method == 'boser':
			from  daal.algorithms.svm.training import boser
			algorithm = prediction.Batch (method=boser, fptype=self.dtype)

		if self.kernel == 'linear':
			algorithm.parameter.kernel = kernel_function.linear.Batch ()
		elif self.kernel == 'rbf':
			algorithm.parameter.kernel = kernel_function.rbf.Batch ()

		algorithm.input.setTable (classifier.prediction.data, testData)
		algorithm.input.setModel (classifier.prediction.model, trainingResult.get (classifier.training.model))
		algorithm.compute ()
		predictionResult = algorithm.getResult()
		predictedResponses = predictionResult.get (classifier.prediction.prediction)
		'''
		block = BlockDescriptor ()
		predictedResponses.getBlockOfRows (0, predictedResponses.getNumberOfRows (), readWrite, block)
		predictArray = block.getArray ()
		predictArray[predictArray < 0] = -1
		predictArray[predictArray >= 0] = 1
		predictedResponses.releaseBlockOfRows (block)
		'''
		return predictedResponses
	'''
	Arguments: serialized numpy array
	Returns Compressed numpy array
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
	Arguments: deserialized numpy array
	Returns decompressed numpy array
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
				compressedData = BinarySVM.compress (self, bufferArray)
				np.save (fileName, compressedData)
			else:
				comBufferArray = BinarySVM.compress (self, bufferArray)
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
			bufferArray = BinarySVM.decompress (self, bufferArray)
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
		self._qualityMetricSetResult = ClassifierQualityMetrics(testGroundTruth, predictResults, 2)
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
	def printAllQualityMetrics(self, qualityMetricSetResult):
	
		# Print the quality metrics
		printNumericTable(qualityMetricSetResult.get('confusionMatrix'), "Confusion matrix:")

		print("Accuracy:      {0:.3f}".format(qualityMetricSetResult.get('accuracy')))
		print("Precision:     {0:.3f}".format(qualityMetricSetResult.get('precision')))
		print("Recall:        {0:.3f}".format(qualityMetricSetResult.get('recall')))
		print("F1-score:      {0:.3f}".format(qualityMetricSetResult.get('fscore')))
		print("Specificity:   {0:.3f}".format(qualityMetricSetResult.get('specificity')))
		print("AUC:           {0:.3f}".format(qualityMetricSetResult.get('auc')))

class MultiSVM:
	'''
	Constructor to set SVM training parameters
	'''
	def __init__(self, nClasses, method="boser", C = 1, tolerence = 0.001, tau = 0.000001, maxIterations = 1000000, cacheSize = 8000000, doShrinking = True, kernel = 'linear',
				 sigma = 0,k=1, b=0,dtype=float64):
		'''
		nClasses: number of classes
		method: 'boser', default: 'boser'
			computation method
		C: deafult: 1
			Upper bound in conditions of the quadratic optimization problem.
		tolerance: default: '0.001'
			Training accuracy/ stopping criteria
		tau: default: 0.000001
			Tau parameter of the WSS scheme.
		maxiterations: default: 1000000
			Maximal number of iterations for the algorithm.
		cacheSize: default: 8000000
			cachesize for storing values of kernal matrix.
		doShringing: True/false, default: True
			flag to set shrinking optimization techinique
		kernel: 'linear'/'rbf', default: 'linear
		k: default: 1
			coefficient value of k when kernal function is 'linear'
		b: 	default: 0
			coeffiecent value of b of linear function
		dtype: intc, float32, float63, intc	
		'''	
		self.method = method
		# Print error message here"
		self.dtype = dtype
		self.C = C
		self.tolerence = tolerence
		self.tau = tau
		self.maxIterations = maxIterations
		self.cachesize = cacheSize
		self.doShrinking = doShrinking
		self.kernel = kernel
		if self.kernel == "rbf":
			self.sigma = sigma
		elif self.kernel == "linear":
			self.k = k
			self.b = b
		self.classes = nClasses
	'''
	Arguments: train data feature values of type nT, train data target values(type nT)
	Returns training results object
	'''
	def training(self, trainData, trainDependentVariables):

		#Set algorithms parameters
		if self.method == 'boser':
			from  daal.algorithms.svm.training import boser
			trainingBatch = training.Batch (method=boser, fptype=self.dtype)
			predictionBatch = prediction.Batch()
	 
		if self.kernel == 'linear':
			trainingBatch.parameter.kernel = kernel_function.linear.Batch (method=boser, fptype=self.dtype)
			trainingBatch.parameter.k = self.k
			trainingBatch.parameter.b = self.b
		elif self.kernel == 'rbf':
			trainingBatch.parameter.kernel =  kernel_function.rbf.Batch(method=boser, fptype=self.dtype)
			trainingBatch.parameter.sigma = self.sigma

		trainingBatch.parameter.cacheSize = self.cachesize
		trainingBatch.parameter.C = self.C
		trainingBatch.parameter.accuracyThreshold = self.tolerence
		trainingBatch.parameter.tau = self.tau
		trainingBatch.parameter.maxIterations = self.maxIterations
		trainingBatch.parameter.doShrinking = self.doShrinking

		algorithm = multi_class_classifier.training.Batch (self.classes)
		algorithm.parameter.training = trainingBatch
		algorithm.parameter.prediction = predictionBatch
		algorithm.input.set (classifier.training.data, trainData)
		algorithm.input.set (classifier.training.labels, trainDependentVariables)
		trainingResult = algorithm.compute ()
		return trainingResult
	'''
	Arguments: training result object, test data feature values(type nT)
	Returns predicted values(type nT)
	'''
	def predict(self, trainingResult, testData): #give other parameters

		if self.method == 'boser':
			from  daal.algorithms.svm.training import boser
			predictionBatch = prediction.Batch (method=boser, fptype=self.dtype)
			trainingBatch = training.Batch (method=boser, fptype=self.dtype)

		if self.kernel == 'linear':
			predictionBatch.parameter.kernel = kernel_function.linear.Batch ()
		elif self.kernel == 'rbf':
			predictionBatch.parameter.kernel = kernel_function.rbf.Batch ()

		algorithm = multi_class_classifier.prediction.Batch (self.classes)
		algorithm.parameter.training = trainingBatch
		algorithm.parameter.prediction = predictionBatch
		algorithm.input.setTable (classifier.prediction.data, testData)
		algorithm.input.setModel (classifier.prediction.model, trainingResult.get (classifier.training.model))
		algorithm.compute ()
		predictionResult = algorithm.getResult()
		predictedResponses = predictionResult.get (classifier.prediction.prediction)
		# Change the predicted values to 1 and -1
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
				compressedData = MultiSVM.compress (self, bufferArray)
				np.save (fileName, compressedData)
			else:
				comBufferArray = MultiSVM.compress (self, bufferArray)
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
			bufferArray = MultiSVM.decompress (self, bufferArray)
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
		self._qualityMetricSetResult = ClassifierQualityMetrics( testGroundTruth, predictResults, self.classes)
		return self._qualityMetricSetResult		
		
	'''
	Arguments: training result object, test data feature values of type nT, test data actual target values(type nT)
	Returns predicted values(type nT), quality metrics object for multi-classifier 
	'''
	def predictWithQualityMetrics(self, trainingResult, testData,testGroundTruth):

		# Retrieve predicted labels

		predictResults = self.predict (trainingResult, testData)
		self._qualityMetricSetResult=self.qualityMetrics(predictResults, testGroundTruth)
		return predictResults, self._qualityMetricSetResult
		

	'''
	Arguments: quality metrics object for multi-class  classifier 
	Prints Accuracy, error rate, Micro precision,Micro recall,Micro F-score,Macro precision,Macro recall,Macro F-score
	'''
	def printAllQualityMetrics(self, qualityMetricSetResult):
		# Print the quality metrics
		printNumericTable(qualityMetricSetResult.get('confusionMatrix'), "Confusion matrix:")

		print ("Average accuracy: {0:.3f}".format (qualityMetricSetResult.get('averageAccuracy')))
		print ("Error rate:       {0:.3f}".format (qualityMetricSetResult.get('errorRate')))
		print ("Micro precision:  {0:.3f}".format (qualityMetricSetResult.get('microPrecision')))
		print ("Micro recall:     {0:.3f}".format (qualityMetricSetResult.get('microRecall')))
		print ("Micro F-score:    {0:.3f}".format (qualityMetricSetResult.get('microFscore')))
		print ("Macro precision:  {0:.3f}".format (qualityMetricSetResult.get('macroPrecision')))
		print ("Macro recall:     {0:.3f}".format (qualityMetricSetResult.get('macroRecall')))
		print ("Macro F-score:    {0:.3f}".format (qualityMetricSetResult.get('macroFscore')))
