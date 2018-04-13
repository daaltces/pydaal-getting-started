import warnings
import operator
import time, datetime
from itertools import takewhile
import numpy as np
from scipy.stats import mode
from sklearn.model_selection import KFold
from daal.algorithms import classifier
from daal.algorithms.kdtree_knn_classification import training, prediction


from daal.algorithms.multi_class_classifier import quality_metric_set as multiclass_quality
from daal.algorithms.classifier.quality_metric import multiclass_confusion_matrix 
from daal.algorithms.svm import quality_metric_set as twoclass_quality
from daal.algorithms.classifier.quality_metric import binary_confusion_matrix
from daal.data_management import (BlockDescriptor_Float64, readOnly,HomogenNumericTable,
                                   InputDataArchive, Compressor_Zlib, level9, CompressionStream)

from collections import namedtuple

# Two-class quality metrics type
TwoClassMetrics = namedtuple('TwoClassMetrics',
		['accuracy', 'precision', 'recall', 'fscore', 'specificity', 'auc'])

# Multi-class quality metrics type
MultiClassMetrics = namedtuple('MultiClassMetrics',
		['accuracy', 'errorRate', 'microPrecision', 'microRecall',
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
	
class _Results:	
	global bestParams 
	def __init__(self,_sortRes ,bestParams,_bestVal):
		self._sortRes = _sortRes
		self.bestParams= bestParams
		self._bestVal=_bestVal
	def viewAllResults(self):#sort this	
		for key, value in self._sortRes.items():
			print ("%s: %s" % (key, value))
	def results(self):		
			return self._sortRes
	def bestResult(self):		#rewrite this better
		_bestVal = list(self._sortRes.values())[0]
		_bestResult = list(takewhile(lambda args: args[1] == _bestVal, self._sortRes.items()))
		bestParams = [x[0] for x in _bestResult ]
		return{'Best Parmeters':bestParams, 'score':_bestVal}
	

class _trainResults(_Results):
	def __init__(self,_sortRes ,bestParams,_bestVal,trainingResult):
		_Results.__init__(self,_sortRes ,bestParams,_bestVal)
		self.trainingResult=trainingResult
	def bestModel(self):
		return self.trainingResult		
			
		
class GridSearch:
	global trainingResult
	def __init__(self,algo,training, prediction, tuned_parameters = None,
					score=None,best_score_criteria='high', 
					create_best_training_model = False,
					save_model=False,nClasses=None ):#cv not used yet
		self.algo = algo
		self.training = training
		self.prediction = prediction
		self.tuned_parameters = tuned_parameters
		if score is None : self.score = 'accuracy'
		else: self.score = score
		self.best_score_criteria= best_score_criteria	
		self.create_best_training_model=create_best_training_model
		self.save_model=save_model
		self.nClasses = nClasses	
		

	'''
	Arguments: Numeric table
	Returns array of numeric table. 
	'''
	def getArrayFromNT(self,nT):
		doubleBlock = BlockDescriptor_Float64()
		firstRow = 0
		lastRow = nT.getNumberOfRows()
		firstCol = 0
		lastCol = nT.getNumberOfColumns()
		nT.getBlockOfRows(firstRow, lastRow, readOnly, doubleBlock)
		getArray = doubleBlock.getArray()
		return getArray
		
	'''
	call the serialize() function to invoke compress() method
	Arguments: serialized numpy array
	Returns Compressed numpy array
	'''

	def compress(self,arrayData):
		compressor = Compressor_Zlib ()
		compressor.parameter.gzHeader = True
		compressor.parameter.level = level9
		comprStream = CompressionStream (compressor)
		comprStream.push_back (arrayData)
		compressedData = np.empty (comprStream.getCompressedDataSize (), dtype=np.uint8)
		comprStream.copyCompressedArray (compressedData)
		return compressedData	
		
	#-------------------
	#***Serialization***
	#-------------------
	'''
	Method 1:
		Arguments: data(type nT/model)
		Returns  dictionary with serailized array (type object) and object Information (type string)
	Method 2:
		Arguments: data(type nT/model), fileName(.npy file to save serialized array to disk)
		Saves serialized numpy array as "fileName" argument
		Saves object information as "filename.txt"
	 Method 3:
		Arguments: data(type nT/model), useCompression = True
		Returns  dictionary with compressed array (type object) and object information (type string)
	Method 4:
		Arguments: data(type nT/model), fileName(.npy file to save serialized array to disk), useCompression = True
		Saves compresseed numpy array as "fileName" argument
		Saves object information as "filename.txt"
	'''

	def serialize(self,data, fileName=None, useCompression= False):
		buffArrObjName = (str(type(data)).split()[1].split('>')[0]+"()").replace("'",'')
		dataArch = InputDataArchive()
		data.serialize (dataArch)
		length = dataArch.getSizeOfArchive()
		bufferArray = np.zeros(length, dtype=np.ubyte)
		dataArch.copyArchiveToArray(bufferArray)
		if useCompression == True:
			if fileName != None:
				if len (fileName.rsplit (".", 1)) == 2:
					fileName = fileName.rsplit (".", 1)[0]
				compressedData = self.compress(bufferArray)
				np.save (fileName, compressedData)
			else:
				comBufferArray = self.compress (bufferArray)
				serialObjectDict = {"Array Object":comBufferArray,
									"Object Information": buffArrObjName}
				return serialObjectDict
		else:
			if fileName != None:
				if len (fileName.rsplit (".", 1)) == 2:
					fileName = fileName.rsplit (".", 1)[0]
				np.save(fileName, bufferArray)
			else:
				serialObjectDict = {"Array Object": bufferArray,
									"Object Information": buffArrObjName}
				return serialObjectDict
		infoFile = open (fileName + ".txt", "w")
		infoFile.write (buffArrObjName)
		infoFile.close ()	
		print("Data successfully serialized and saved as {} and {}".format(fileName,infoFile.name))
	
	def _predict(self,testData, trainingResult):
		
		self._predAlgorithm .input.setModel(classifier.prediction.model, 
												trainingResult.get(classifier.training.model))
		self._predAlgorithm .input.setTable(classifier.prediction.data, 
												testData)	#check if this is table everywhere		
		_predictionResult = self._predAlgorithm.compute()	
		return _predictionResult
	
	def _predictParams(self,keys,values):
		
		for i in range (len (keys)):
			if self.algo_name=='gbt':
				if values[i] in dir(self.algo.prediction):				
					self._predAlgorithm.parameter().__setattr__ (keys[i],getattr(self.algo.prediction,values[i]))
				elif values[i] in dir(self.algo):#this is an issue in knn,dt so this condition	
					self._predAlgorithm.parameter().__setattr__ (keys[i],getattr(self.algo,values[i]))
				else:			
					self._predAlgorithm.parameter().__setattr__ (keys[i], values[i])	
			
			else:
				if values[i] in dir(self.algo.prediction):				
					self._predAlgorithm.parameter.__setattr__ (keys[i],getattr(self.algo.prediction,values[i]))
				elif values[i] in dir(self.algo):#this is an issue in knn,dt so this condition	
					self._predAlgorithm.parameter.__setattr__ (keys[i],getattr(self.algo,values[i]))
				else:			
					self._predAlgorithm.parameter.__setattr__ (keys[i], values[i])						
	
	def _train(self,keys,values,trainData,trainDependentVariables,pruneData,pruneGroundTruth):	
		self._algorithm.input.set (classifier.training.data, 
											trainData)
		self._algorithm.input.set (classifier.training.labels, 
											trainDependentVariables)
		if self.algo_name =="decision_tree":
			if pruneData !=None and pruneGroundTruth!=None:	
				self._algorithm.input.setTable(self.training.dataForPruning, 
											pruneData)
				self._algorithm.input.setTable(self.training.labelsForPruning, 
										pruneGroundTruth)
		for i in range (len (keys)):
			if self.algo_name=='gbt':
				if values[i] in dir(self.algo.training): #this is needed for decision forest	
					self._algorithm.parameter().__setattr__ (keys[i],getattr(self.algo.training,values[i]))
				elif values[i] in dir(self.algo):#this is an issue in knn,dt so this condition	
					self._algorithm.parameter().__setattr__ (keys[i],getattr(self.algo,values[i]))
				else:			
					self._algorithm.parameter().__setattr__ (keys[i], values[i])	
			else:
				if values[i] in dir(self.algo.training):	
					self._algorithm.parameter.__setattr__ (keys[i],getattr(self.algo.training,values[i]))
				elif values[i] in dir(self.algo):#this is an issue in knn,dt so this condition	
					self._algorithm.parameter.__setattr__ (keys[i],getattr(self.algo,values[i]))
				else:			
					self._algorithm.parameter.__setattr__ (keys[i], values[i])		
		trainingResult= self._algorithm.compute ()
		return trainingResult
				
	def _sortResults(self):	
		_sortRes={}
		if self.best_score_criteria == 'low':
			for key,value in  sorted(self.results.items(), key=operator.itemgetter(1)):	
				_sortRes[key]=value
		else:
			for key,value in  sorted(self.results.items(), key=operator.itemgetter(1), reverse = True):
				_sortRes[key]=value
		return _sortRes		

	def train(self, full_trainData, full_trainDependentVariables,pruneData=None,pruneLables=None, splits=2):		
		
		dict_TrainData = full_trainData.getDictionary()		
		#Create combinations of params
		import itertools as it
		combinations = it.product (*(sorted(self.tuned_parameters[0][key]) for key in self.tuned_parameters[0]))
		keys = list (self.tuned_parameters[0].keys ())	
		if len(self.tuned_parameters)==2:
			combinations_ = it.product (*(sorted(self.tuned_parameters[1][key]) for key in self.tuned_parameters[1]))		
			keys_=list(self.tuned_parameters[1].keys ())	
			
		self.algo_name = self.training.__name__.split('.')[2]
	
		#Split data using KFold
		kf = KFold(n_splits=splits)
		kf.get_n_splits(self.getArrayFromNT(full_trainData))		
		full_trainIDV_array = self.getArrayFromNT(full_trainData)
		full_trainDV_array = self.getArrayFromNT(full_trainDependentVariables)		
		#use the set and get dictionary here
		try:			
			self._algorithm = self.training.Batch(self.nClasses)				
		except:
			self._algorithm = self.training.Batch()
		try:	
			self._predAlgorithm = self.prediction.Batch(self.nClasses)	
		except:	
			self._predAlgorithm = self.prediction.Batch()	
		if self.algo_name=='gbt':
			self._algorithm.parameter=self._algorithm.parameter()
		if len(set(keys) - set(dir(self._algorithm.parameter))) >0:	
			warnings.warn ('Error: Invalid parameters')
			raise SystemExit		
		_algoCloned = self._algorithm.clone ()
		_predAlgoClone =self._predAlgorithm.clone()		
		_tempD={}
		self.results = {}		
		for values in combinations:		
			_temp=[]			
			for train_index, test_index in kf.split(full_trainIDV_array):
				self._algorithm = _algoCloned.clone()
				self._predAlgorithm=_predAlgoClone.clone()
				trainData,testData  = HomogenNumericTable(full_trainIDV_array[train_index]), \
														HomogenNumericTable(full_trainIDV_array[test_index])
				trainData.setDictionary(dict_TrainData)										
				testData.setDictionary(dict_TrainData)										
				trainDependentVariables, testGroundTruth = HomogenNumericTable(full_trainDV_array[train_index]), \
														HomogenNumericTable(full_trainDV_array[test_index])	
				trainingResult = self._train(keys,values,trainData,trainDependentVariables,pruneData,pruneLables)	
				if len(self.tuned_parameters)==2:		
					combinations_ = it.product (*(sorted(self.tuned_parameters[1][key]) for key in self.tuned_parameters[1]))		
					for values_ in combinations_:
						self._predictParams(keys_,values_)	
						_predictionResult = self._predict(testData,trainingResult)
						if self.algo_name=='gbt':
							_met = ClassifierQualityMetrics(testGroundTruth,_predictionResult.
								get(classifier.prediction.prediction),
								nclasses=self._algorithm.parameter().__getattr__ ('nClasses'))
						else:
							_met = ClassifierQualityMetrics(testGroundTruth,_predictionResult.
								get(classifier.prediction.prediction),
								nclasses=self._algorithm.parameter.__getattr__ ('nClasses'))								
						try:							
							_tempD[str(dict(zip(keys+keys_,values+values_)))].append(_met.get(self.score))							
						except KeyError:							
							_tempD[str(dict(zip(keys+keys_,values+values_)))]=[_met.get(self.score)]												
				else:
					_predictionResult = self._predict(testData,trainingResult)
					_met = ClassifierQualityMetrics(testGroundTruth,_predictionResult.
									get(classifier.prediction.prediction),
									nclasses=self._algorithm.parameter.__getattr__ ('nClasses'))
					_temp.append(_met.get(self.score))
			if len(self.tuned_parameters)==1:				
				self.results[str(dict(zip(keys,values)))]= np.mean(_temp)
		if 	len(self.tuned_parameters)==2:
			self.results = {key:np.mean(value) for (key,value) in _tempD.items()}
		_sortRes = self._sortResults()
		_bestVal = list(_sortRes.values())[0]
		_bestResult = list(takewhile(lambda args: args[1] == _bestVal, _sortRes.items()))
		bestParams = [x[0] for x in _bestResult]
		if self.create_best_training_model == True:
			self._algorithm = _algoCloned.clone()
			trainingResult_ = self._train(list(eval(bestParams[0]).keys()),list(eval(bestParams[0]).values()),full_trainData, full_trainDependentVariables,pruneData,pruneLables)	
			if self.save_model==True:
				self.serialize(trainingResult_, fileName="trainRes-{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now()))
			return _trainResults(_sortRes ,bestParams,_bestVal, trainingResult_)
		return _Results(_sortRes ,bestParams,_bestVal)
			

				
		