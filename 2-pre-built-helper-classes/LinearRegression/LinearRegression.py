
from daal.algorithms.linear_regression import training, prediction
from daal.data_management import InputDataArchive, OutputDataArchive
from daal.data_management import Compressor_Zlib, Decompressor_Zlib, level9, DecompressionStream, CompressionStream, HomogenNumericTable
from daal.data_management import BlockDescriptor, readWrite

import daal.algorithms.linear_regression.quality_metric_set as quality_metric_set
from daal.algorithms.linear_regression.quality_metric import single_beta, group_of_betas

from utils import printNumericTable
import numpy as np
from numpy import float32, float64
import warnings


class LinearRegression:
    '''
    Constructor to set Linear Regression training parameters
	parameters: 
	method: 'normeq'/'qr', default: 'normeq'
			used to decide the calculation method. 'normeq' is normal equation,  'qr' is QR decomposition
			
	interceptFlag: True/False, default:	'True'
			decides whether or not intercept component to be evaluated
	dtype: intc/float32, float64, default: float64		
    '''
    def __init__(self, method = 'normEq', interceptFlag = True, dtype = float64):
        self.method = method
        #Print error message here"
        self.interceptFlag = interceptFlag
        self.dtype = dtype
    '''
    Arguments: train data feature values(type nT), train data target values(type nT)
    Returns training results object
    '''
    def training(self, trainData, trainDependentVariables):

        if self.method == 'normEq':
            method = training.normEqDense
        elif self.method == 'qr':
            method = training.qrDense
        else:
            warnings.warn ('Invalid method, using default dense Normal Equation method')
            method = training.normEqDense
        algorithm = training.Batch(method=method, fptype = self.dtype)

        # Pass a training data set and dependent values to the algorithm
        algorithm.input.set (training.data, trainData)
        algorithm.input.set (training.dependentVariables, trainDependentVariables)
        algorithm.parameter.interceptFlag = self.interceptFlag

        # Build linear regression model and retrieve the algorithm results
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

        # Predict values of multiple linear regression and retrieve the algorithm results
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
                compressedData = LinearRegression.compress (self,bufferArray)
                np.save (fileName, compressedData)
            else:
                comBufferArray = LinearRegression.compress (self,bufferArray)
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
            bufferArray = LinearRegression.decompress (self,bufferArray)
        dataArch = OutputDataArchive (bufferArray)
        try:
            deSerialObj = eval (buffArrObjName)
        except AttributeError:
            deSerialObj = HomogenNumericTable ()
        deSerialObj.deserialize (dataArch)
        return deSerialObj
    '''
    Arguments: training result object, test data feature values of type nT, number of dependent variables, insignificant beta indexes (type list/int)
    Returns  reduced model predicted values (type nT)
    '''
    def predictReducedModelResults(self,trainingResult,trainData, nDependentVariables, reducedBeta):
        model = trainingResult.get(training.model)
        betas = model.getBeta ()
        nBetas = model.getNumberOfBetas ()
        savedBeta = np.zeros((nDependentVariables,nBetas))
        block = BlockDescriptor ()
        betas.getBlockOfRows (0, nDependentVariables, readWrite, block)
        pBeta = block.getArray()
        if type (reducedBeta) == int: reducedBeta = [reducedBeta]
        for beta in reducedBeta:
            for i in range (nDependentVariables):
                savedBeta[i][beta] = pBeta[i][beta]
                pBeta[i][beta] = 0
        betas.releaseBlockOfRows (block)
        printNumericTable(betas)
        predictedResults = LinearRegression.predict(self,trainingResult,trainData)
        block = BlockDescriptor ()
        betas.getBlockOfRows (0, nBetas, readWrite, block)
        pBeta = block.getArray()
        for beta in reducedBeta:
            for i in range (0, nDependentVariables):
                 pBeta[i][beta] = savedBeta[i][beta]
        betas.releaseBlockOfRows (block)
        printNumericTable(betas)
        return predictedResults
    '''
    Arguments: training result object, test data feature values of type nT, test data actual target values(type nT), insignificant beta indexes (type list/int)
    Returns predicted values(type nT), reduced model predicted values (type nT), single beta metrics result, group beta metrics result
    '''
    def predictWithQualityMetrics(self, trainingResult, testData, testDependentVariables, reducedBetaIndex = None):
        predictResults = LinearRegression.predict(self,trainingResult,testData)
        model =trainingResult.get(training.model)
        numberOfBetas = model.getNumberOfBetas()
        if reducedBetaIndex != None:
            predictedReducedModelResults = LinearRegression.predictReducedModelResults (self,trainingResult,testData, testDependentVariables.getNumberOfColumns(),reducedBetaIndex)
            redBetas = len(reducedBetaIndex)
        else:
            predictedReducedModelResults = predictResults
            redBetas = 1
        qualityMetricSet = quality_metric_set.Batch(model.getNumberOfBetas(),model.getNumberOfBetas()-redBetas)
        
        singleBeta = single_beta.Input.downCast (qualityMetricSet.getInputDataCollection ().getInput (quality_metric_set.singleBeta))
        singleBeta.setDataInput (single_beta.expectedResponses, testDependentVariables)
        singleBeta.setDataInput (single_beta.predictedResponses, predictResults)
        singleBeta.setModelInput (single_beta.model, model)

        # Set input for a group of betas metrics algorithm
        groupOfBetas = group_of_betas.Input.downCast (qualityMetricSet.getInputDataCollection ().getInput (quality_metric_set.groupOfBetas))
        groupOfBetas.set (group_of_betas.expectedResponses, testDependentVariables)
        groupOfBetas.set (group_of_betas.predictedResponses, predictResults)
        groupOfBetas.set (group_of_betas.predictedReducedModelResponses,predictedReducedModelResults)

        # Compute quality metrics0
        qualityMetricSet.compute ()

        # Retrieve the quality metrics
        qmsResult = qualityMetricSet.getResultCollection ()
        resultSingleBeta = single_beta.Result.downCast(qmsResult.getResult(quality_metric_set.singleBeta))
        resultGroupBeta = group_of_betas.Result.downCast (qmsResult.getResult (quality_metric_set.groupOfBetas))
        return predictResults, predictedReducedModelResults, resultSingleBeta, resultGroupBeta
    '''
    Arguments: single beta metrics result, group beta metrics result
    Prints RMSE, variance, z-score statistic, confidenceIntervals, inverseOfXtX matrix, variance-covariance matrix, expectedMean, expectedVariance, SSR, SST, R-square, f-statistic
    '''
    def printAllQualityMetrics(self, resultSingleBeta, resultGroupBeta):
        print ("Quality metrics for a single beta")
        printNumericTable (resultSingleBeta.getResult (single_beta.rms),
                           "Root means square errors for each response (dependent variable):")
        printNumericTable (resultSingleBeta.getResult (single_beta.variance), "Variance for each response (dependent variable):")
        printNumericTable (resultSingleBeta.getResult (single_beta.zScore), "Z-score statistics:")
        printNumericTable (resultSingleBeta.getResult (single_beta.confidenceIntervals),
                           "Confidence intervals for each beta coefficient:")
        printNumericTable (resultSingleBeta.getResult (single_beta.inverseOfXtX), "Inverse(Xt * X) matrix:")

        coll = resultSingleBeta.getResultDataCollection (single_beta.betaCovariances)
        for i in range (0, coll.size ()):
            message = "Variance-covariance matrix for betas of " + str (i) + "-th response"
            betaCov = resultSingleBeta.get (single_beta.betaCovariances, i)
            printNumericTable (betaCov, message)

        # Print quality metrics for a group of betas
        print ("Quality metrics for a group of betas")
        printNumericTable (resultGroupBeta.get (group_of_betas.expectedMeans),
                           "Means of expected responses for each dependent variable:", 0, 0, 20)
        printNumericTable (resultGroupBeta.get (group_of_betas.expectedVariance),
                           "Variance of expected responses for each dependent variable:", 0, 0, 20)
        printNumericTable (resultGroupBeta.get (group_of_betas.regSS), "Regression sum of squares of expected responses:", 0, 0,
                           20)
        printNumericTable (resultGroupBeta.get (group_of_betas.resSS),
                           "Sum of squares of residuals for each dependent variable:", 0, 0, 20)
        printNumericTable (resultGroupBeta.get (group_of_betas.tSS), "Total sum of squares for each dependent variable:", 0, 0,
                           20)
        printNumericTable (resultGroupBeta.get (group_of_betas.determinationCoeff),
                           "Determination coefficient for each dependent variable:", 0, 0, 20)
        printNumericTable (resultGroupBeta.get (group_of_betas.fStatistics), "F-statistics for each dependent variable:", 0, 0,
                           20)

