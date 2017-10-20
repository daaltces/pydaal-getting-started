
Pre-built helper classes are constructed using PyDAAL and usage example(s) are provided in each folder.<br>

Generic methods handled in each class are:
1. Training: fits the data to create a model
2. Prediction: uses training results to predict responses
3. Quality Metrics: evaluates model quality based on the type of model
4. Model Storage and Portability: enables model storage and retrieval with serialization and compression

## What are the files in this folder?
Each folder in 2-pre-built-helper-class contains:
1. A pre-built helper class. 
2. Related usage example(s) for practical purpose implementing all methods of the pre-built helper class. 
3. Documentation on methods, parameters and possible arguments.


##  How do I use this class?
Download 2-pre-built-helper-class/\<*algorithms*> in your working directory. Import the respective class and start building your models<br>

*For example:* To build a linear regression model 

1. Download 2-pre-built-helper-class/LinearRegression
2. Import the class file LinearRegression.py in your python program<br>
```python
from LinearRegression import LinearRegression
```
3. Create LinearRegression object and start building your model

```python
#Instantiate Linear Regression object
lr = LinearRegression()
#Training
trainingResults = lr.training(trainData, trainDependentVariables)
#Prediction	
predResponses = lr.prediction(trainingResults, testData)
#Serialize
lr.serialize(trainingResults, fileName = "storeTrainingResults.npy")
#Deserialize
retrieveTrainingResults = lr.deserialize(fileName = "storeTrainingResults.npy")
#Predict with Metrics
predRes, predResRed, singleBeta, groupBeta = lr.predictWithQualityMetrics(trainingResult, trainData,trainDependentVariables,reducedBetaIndex=[2,10])
#Print Metrics results
lr.printAllQualityMetrics(singleBeta,groupBeta)
```

For practice, Run the examples from 2-pre-built-helper-class/\<*algorithm*>/\<*usage example*>.
## Do the usage examples cover all possibilities of using these methods?

No. These examples may not cover all possible approaches, kindly refer to the documentation provided in 2-pre-built-helper-class/\<*algorithm*>/\<*documentation*> or the comment section above each method in the class file.
