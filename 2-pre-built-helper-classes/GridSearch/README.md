
GridSearch helper class can be used to tune hyper parameters of Intel DAAL algorithms. This helper class can be used on all the existing classification algorithms. Usage examples of all the algorithms are also available for practice. <br>

## What are the current functionalities supported by this helper class?<br>
a.	Split the data using KFold cross-validation technique and  generates data splits<br>
b.	Train  the data for all combinations of hyperparameters<br>
c.	Evaluate model performance for each hyperparameter combination<br>
d.	Return the hyperparameter combination with the best score<br>
e.   Return the training model computed using the best hyperparameter combination<br>
f.  Serialize and save the best trained model result<br>

## Usage<br>
### Syntax
```python
GridSearch(**args, tuned_parameters = None, score='accuracy',
			best_score_criteria='high',
			create_best_training_model = False,
			save_model=False,nClasses=None )	
```
**args for all the classifiers are daal *algorithm*, *training* and *prediction* classes

### Example 

#### Instantiate a grid search object
```python
from GridSearch import GridSearch
import   daal.algorithms.svm as svm
from  daal.algorithms.svm import training, prediction

#create a dictionary of hyperparameter values in a list
svm_params = [{'C':[0.5,1],
				'accuracyThreshold':[0.01,0.001]]   
#Create GridSearch object                
clf = GridSearch(svm,training,prediction, 
				tuned_parameters = svm_params,score='accuracy',
				best_score_criteria='high',
				create_best_training_model=False,
				save_model=False,nClasses=None)
```                
#### Train models for all the hyperparameter combinations
```python
result = clf.train(trainData,trainGroundTruth,pruneData=None, pruneLables=None, splits=2)
#prune data and labels are used for Decision Tree Algorithm
```
#### View all the results / Print the best result
```python   
result.viewAllResults()
print(result.bestResult())
```                

