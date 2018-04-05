import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','python','source'))
from daal.algorithms import pca
from daal.algorithms.pca import transform
from daal.algorithms import covariance as covariance_
from daal.algorithms.normalization import zscore, minmax
import daal.algorithms.pca.quality_metric_set as quality_metric_set
from daal.algorithms.pca.quality_metric import explained_variance
from daal.data_management import InputDataArchive, OutputDataArchive
from daal.data_management import Compressor_Zlib, Decompressor_Zlib, level9, DecompressionStream, CompressionStream, HomogenNumericTable
import numpy as np
from numpy import float32, float64
import warnings
from customUtils import getArrayFromNT


class PCA:
	'''
	Constructor to set PCA training parameters
	parameters: 
	method: 'defaultDense'/'svdDense', default: 'defaultDense'
			used to decide the calculation method. 'defaultDense' is covariance
	covariance: (applicable only if method='defaultDense'), default : 'defaultDense' 	
		available values - singlePassDense, sumDense, fastCSR, singlePassCSR, sumCSR
	normalization :  (applicable only if method='svdDense')default: 'zscore'
		available values - zscore, minmax
	nComponents: default 0
		if 0 number of components is number of features
	isDeterministic: default: False	
		If true, the algorithm applies the "sign flip" technique to the results.
	resultsToCompute: default: None
		The 64-bit integer flag that specifies which optional result to compute.
		Provide one of the following values to request a single characteristic or use bitwise OR to request a combination of the characteristics:	 				 
		mean 
		variance
		eigenvalue	
	'''
	def __init__(self, method = 'defaultDense',covariance = "defaultDense", normalization="zscore", nComponents = 0,isDeterministic = False,fptype=float64, resultsToCompute = 0):

		if method == 'defaultDense' or pca.defaultDense:
			self.method = pca.defaultDense
		elif method == 'svdDense' or pca.svdDense:
			self.method = pca.svdDense
		else:
			warnings.warn ('Invalid method, using default dense (correlation) method')
			self.method = pca.defaultDense

		if self.method == pca.defaultDense:
			if covariance =="defaultDense" or covariance_.defaultDense:
				self.covariance = covariance_.defaultDense
			elif covariance == "singlePassDense" or covariance_.singlePassDense:
				self.covariance = covariance_.singlePassDense
			elif covariance == "sumDense" or covariance_.sumDense:
				self.covariance = covariance_.sumDense
			elif covariance == "fastCSR" or covariance_.fastCSR:
				self.covariance = covariance_.fastCSR
			elif covariance == "singlePassCSR" or covariance_.singlePassCSR:
				self.covariance = covariance_.singlePassCSR
			elif covariance == "sumCSR" or covariance_.sumCSR:
				self.covariance = covariance_.sumCSR
			else:
				warnings.warn ('Invalid covariance method, using default dense method')
				self.covariance = covariance_.defaultDense
		else:
			if normalization == "zscore" or zscore:
				self.normalization = zscore
			elif normalization =="minmax" or minmax:
				self.normalization=minmax
			else:
				warnings.warn ('Invalid normalization method, using default dense (zcore) method')
				self.covariance = self.normalization = zscore

		self.nComponents = nComponents
		self.isDeterministic = isDeterministic
		self.dtype = fptype
		self.resultsToCompute=resultsToCompute

	'''
	Arguments: train data feature values(type nT), train data target values(type nT)
	Returns training results object
	'''
	def compute(self, data, corrDataFlag=False):
			
		algorithm = pca.Batch(method=self.method, nComponents =self.nComponents,isDeterministic=self.isDeterministic, fptype = self.dtype)	
		algorithm.parameter.nComponents=self.nComponents
		algorithm.parameter.resultsToCompute=self.resultsToCompute
		if corrDataFlag==False:
			algorithm.input.setDataset (pca.data, data)
		else:
			algorithm.input.setCorrelation (pca.data, data)
		pcaResult = algorithm.compute()
		return pcaResult
	'''
	Arguments: computed pca results,
	returns dictionary with eigenvectors, eigenvalues, means, variances, dataForTransform
	'''
	def getResults(self, pcaResult):
		return {"eigenvectors":pcaResult.get (pca.eigenvectors),
				"eigenvalues" : pcaResult.get (pca.eigenvalues),
				"means":pcaResult.get (pca.means),
				"variances":pcaResult.get (pca.variances),
				"dataForTransform":pcaResult.get (pca.dataForTransform)}
	'''
	Arguments: computed pca results, number of priciple components, data to be transformed
	Returns transformed data
	'''
	def transform(self, pcaResult,data, nComponents=None, useDataForTransformation = False):
		if nComponents== None:
			transformAlgorithm = transform.Batch(fptype = self.dtype)
			transformAlgorithm.parameter.nComponents = self.nComponents
		else:
			transformAlgorithm = transform.Batch(fptype = self.dtype)
			transformAlgorithm.parameter.nComponents = nComponents
		transformAlgorithm.input.setTable(transform.data, data)
		transformAlgorithm.input.setTable(transform.eigenvectors, pcaResult.get(pca.eigenvectors))
		if useDataForTransformation==True:
			transformAlgorithm.input.setTable (transform.dataForTransform, pcaResult.get(pca.dataForTransform))
		transformResult = transformAlgorithm.compute ()		
		return (transformResult.get (transform.transformedData))

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
				compressedData = PCA.compress (self,bufferArray)
				np.save (fileName, compressedData)
			else:
				comBufferArray = PCA.compress (self,bufferArray)
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
			bufferArray = PCA.decompress (self,bufferArray)
		dataArch = OutputDataArchive (bufferArray)
		try:
			deSerialObj = eval (buffArrObjName)
		except AttributeError:
			deSerialObj = HomogenNumericTable ()
		deSerialObj.deserialize (dataArch)
		return deSerialObj
	'''
	Arguments: computed pca result
	Returns a dictionary with explainedVariance,explainedVarianceRatio and noiseVariance
	'''
	def qualityMetrics(self,pcaResult):
		# Create a quality metric set object to compute quality metrics of the PCA algorithm
		qualityMetricSet = quality_metric_set.Batch (self.nComponents)
		explainedVariances = explained_variance.Input.downCast (
			qualityMetricSet.getInputDataCollection ().getInput (quality_metric_set.explainedVariancesMetrics))
		explainedVariances.setInput (explained_variance.eigenvalues, pcaResult.get(pca.eigenvalues))
		# Compute quality metrics
		qualityMetricSet.compute ()
		# Retrieve the quality metrics
		qmsResult = qualityMetricSet.getResultCollection ()
		result = explained_variance.Result.downCast (
				qmsResult.getResult (quality_metric_set.explainedVariancesMetrics))
		qualMet = {"explainedVariance": result.getResult(explained_variance.explainedVariances),
					"explainedVarianceRatio": result.getResult(explained_variance.explainedVariancesRatios),
				   "noiseVariance" : result.getResult(explained_variance.noiseVariance)}
		return qualMet


