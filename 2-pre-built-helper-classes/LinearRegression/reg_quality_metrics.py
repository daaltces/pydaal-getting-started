
""" A class for regression quality metrics."""

import daal.algorithms.linear_regression.quality_metric_set as quality_metric_set
from daal.algorithms.linear_regression.quality_metric import single_beta, group_of_betas

SingleBetaMetrics =['rms', 'variance', 'zScore', 'confidenceIntervals', 'inverseOfXtX', 'betaCovariances']


GroupBetaMetrics = ['expectedMeans', 'expectedVariance', 'regSS', 'resSS',
		 'tSS', 'determinationCoeff', 'fStatistics']


class RegressionQualityMetrics:


	def __init__(self, truth, predictResults,trainedModel,predictedReducedModelResults=None,noReducedBetas=1):
		"""Initialize class parameters

		Args:
		   truth: ground truth
		   predictions: predicted labels
		   nclasses: number of classes
		"""
		if predictedReducedModelResults == None:
			self._predictedReducedModelResults=predictResults
		else:
			self._predictedReducedModelResults =predictedReducedModelResults
		self._truth = truth
		self._predictResults = predictResults
		self._redBetas= noReducedBetas
		self._trainedModel = trainedModel
		self._compute()


	def get(self, metric):
		"""Get a metric from the quality metrics collection

		Args:
		   metric: name of the metric to return

		Returns:
		   A numeric value for the given metric	
		"""
		if metric not in SingleBetaMetrics and metric not in GroupBetaMetrics:
			print("Invalid quality metric: " + metric)		
			return
		if metric in SingleBetaMetrics:			
			if metric =='betaCovariances':
				betaCov=[]
				for i in range(self._predictResults.getNumberOfColumns()):
					betaCov.append(self.resultSingleBeta.get (single_beta.betaCovariances, i))
				return betaCov
			return(self.resultSingleBeta.getResult(getattr(single_beta,metric)))		
		if metric in GroupBetaMetrics:
			return(self.resultGroupBeta.get(getattr(group_of_betas,metric)))
			

	def _compute(self):
		qualityMetricSet = quality_metric_set.Batch(self._trainedModel.getNumberOfBetas(),self._trainedModel.getNumberOfBetas()-self._redBetas)        
		self._singleBeta = single_beta.Input.downCast (qualityMetricSet.getInputDataCollection().getInput (quality_metric_set.singleBeta))
		self._singleBeta.setDataInput (single_beta.expectedResponses, self._truth)
		self._singleBeta.setDataInput (single_beta.predictedResponses, self._predictResults)
		self._singleBeta.setModelInput (single_beta.model, self._trainedModel )
		self._groupOfBetas = group_of_betas.Input.downCast (qualityMetricSet.getInputDataCollection ().getInput (quality_metric_set.groupOfBetas))
		self._groupOfBetas.set (group_of_betas.expectedResponses, self._truth)
		self._groupOfBetas.set (group_of_betas.predictedResponses, self._predictResults)
		self._groupOfBetas.set (group_of_betas.predictedReducedModelResponses,self._predictedReducedModelResults)

		# Compute quality metrics
		qualityMetricSet.compute ()
		# Retrieve the quality metrics
		qmsResult = qualityMetricSet.getResultCollection ()
		self.resultSingleBeta = single_beta.Result.downCast(qmsResult.getResult(quality_metric_set.singleBeta))
		self.resultGroupBeta = group_of_betas.Result.downCast (qmsResult.getResult (quality_metric_set.groupOfBetas))
			
			
			




