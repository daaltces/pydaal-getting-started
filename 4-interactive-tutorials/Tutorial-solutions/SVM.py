   def predict(self, model, testdata):
        """Make predictions for unseen data using a learned model.

        Args:
            model: a learned SVM model
            testdata: new data

        Returns:
            A NumericTable containing predicted labels
        """

        # Create a multiclass classifier object based on the
        # SVM two-class classifier
        #
        # YOUR CODE HERE
        #
        # The multi-class prediction algorithm you need is Batch_Float64MultiClassClassifierWuOneAgainstOne
        # Follow the example in the `train` method to set parameters, including nClasses, and training 
        # and prediction algorithms for the underlying two-class classifier. 
        multiclass_prediction_alg = multiclass_prediction.Batch_Float64MultiClassClassifierWuOneAgainstOne(self._nclasses)
        
        multiclass_prediction_alg.parameter.training = self._svm_training_alg
        multiclass_prediction_alg.parameter.prediction = self._svm_prediction_alg
        # Pass a model and input data
        #
        # YOUR CODE HERE
        #
        # Use the input.setModel method to specify a pre-trained model. The input ID to use is
        # prediction_params.model.
        # Use the input.setTable method to specify test data. The input ID to use is prediction_params.data

        multiclass_prediction_alg.input.setTable(prediction_params.data, testdata) 
        multiclass_prediction_alg.input.setModel(prediction_params.model, model) 

        # Compute and return prediction results
        #
        # YOUR CODE HERE
        results = multiclass_prediction_alg.compute()
    
        #
        # Call the `compute` method of the multi-class prediction algorithm. Store the return value into 
        # variable `results`.
     
        return results.get(prediction_params.prediction)