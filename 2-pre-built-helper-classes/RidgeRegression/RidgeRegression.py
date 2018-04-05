


from daal.algorithms.ridge_regression import training, prediction
from daal.data_management import InputDataArchive, OutputDataArchive
from daal.data_management import Compressor_Zlib, Decompressor_Zlib, level9, DecompressionStream, CompressionStream, HomogenNumericTable
from daal.data_management import BlockDescriptor, readWrite
import numpy as np
from numpy import float32, float64
import warnings


class RidgeRegression:
	'''
	Constructor to set Ridge Regression training parameters
	parameters: 
	ridgeParameters: int/list/numpy array, default: 1
			All dependent variables will have the same lamba value if one value is given
			A list/numpy array can also be given with different values for all dependent variables.

	method: 'defualtDense'/'qrDense', default: 'defaultDense'
			used to decide the calculation method. 'defaultDense' is normal equation
			
	interceptFlag: True/False, default:	'True'
			decides whether or not intercept component to be evaluated
	dtype: intc/float32, float64, default: float64		
	'''
	def __init__(self, ridgeParameters = 1, method = 'defaultDense',  dtype = float64):
		self.method = method
		self.dtype = dtype
		self.ridgeParameters = ridgeParameters
	'''
	Arguments: train data feature values(type nT), train data target values(type nT)
	Returns training results object
	'''
	def training(self, trainData, trainDependentVariables):

		if self.method == 'defaultDense':
			method = training.normEqDense
		else:
			warnings.warn ('Invalid method, using default dense Normal Equation method')
			method = training.normEqDense
		if type(self.ridgeParameters) is list:
			if len(self.ridgeParameters) == trainDependentVariables.getNumberOfRows():
				self.ridgeParameters = np.array(self.ridgeParameters, ndmin=2)
			else:
				warnings.warn ('no. of ridgeParameters must be equal to no. of dependent variables')
				raise SystemExit
		elif type(self.ridgeParameters) is int or type(self.ridgeParameters) is float:
			self.ridgeParameters = np.array(self.ridgeParameters,ndmin=2)
		elif type(self.ridgeParameters) is np.ndarray:
			pass
		else:
			warnings.warn ('Invalid aplha type. ridgeParameters must be type int or float or list')
			raise SystemExit
		nT_ridgeParams = HomogenNumericTable(self.ridgeParameters)		
		algorithm = training.Batch(method=method, fptype = self.dtype)
		algorithm.parameter.ridgeParameters = nT_ridgeParams
		algorithm.input.set (training.data, trainData)
		algorithm.input.set (training.dependentVariables, trainDependentVariables)
		# Build Ridge regression model and retrieve the algorithm results
		trainingResult = algorithm.compute()
		return trainingResult
	'''
	Arguments: training result object, test data feature values(type nT)
	Returns predicted values of type nT
	'''
	def predict(self, trainingResult, testData):
		algorithm = prediction.Batch(fptype = self.dtype)
		# Pass a testing data set and the trained model to the algorithm
		algorithm.input.setTable(prediction.data, testData)
		algorithm.input.setModel(prediction.model, trainingResult.get(training.model))

		# Predict values of multiple Ridge regression and retrieve the algorithm results
		predictionResult = algorithm.compute()
		return (predictionResult.get (prediction.prediction))
	'''
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
				if len(fileName.rsplit (".", 1))==2:
					fileName = fileName.rsplit (".", 1)[0]
				compressedData = RidgeRegression.compress (self,bufferArray)
				np.save (fileName, compressedData)
			else:
				comBufferArray = RidgeRegression.compress (self,bufferArray)
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
	Returns deserialized/ decompressed numeric table/model
	Input can be serialized/ compressed numpy array or serialized/ compressed .npy file saved to disk
	'''

	def deserialize(self,serialObjectDict=None, fileName=None, useCompression=False):
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
			bufferArray = RidgeRegression.decompress (self,bufferArray)
		dataArch = OutputDataArchive (bufferArray)
		try:
			deSerialObj = eval (buffArrObjName)
		except AttributeError:
			deSerialObj = HomogenNumericTable ()
		deSerialObj.deserialize (dataArch)
		return deSerialObj
	

