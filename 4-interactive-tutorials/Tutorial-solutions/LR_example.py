   def predict(self, model, testdata, intercept = True):
        """Make prediction for unseen data using a trained model

        Args:
            model: A trained model
            testdata: New data
            intercept: A boolean to inidicate if intercept needs to be computed 

        Returns:
            A NumericTable containing predicted responses 
        """

        # Create a prediction algorithm object
        #
        # YOUR CODE HERE
        #
        # The algorithm class you want to use is ridge_prediction.Batch
        algorithm = ridge_prediction.Batch()      
       
        
        # Set input
        #
        # YOUR CODE HERE
        #
        # There are two pieces of input to be set: a pre-trained model and input data. You should
        # use the 'input.setModel' and the 'input.setTable' member methods of the
        # algorithm object. The input IDs to use are 'ridge_prediction.model' and 'ridge_prediction.data'
        # respectively.
        algorithm.input.setTable(ridge_prediction.data, testdata)
        algorithm.input.setModel(ridge_prediction.model,model)
     
        # Compute
        #
        # YOUR CODE HERE
        #
        # Call the 'compute' method of your algorithm object, and store the result to 'results'.
        results = algorithm.compute()
        return results.get(ridge_prediction.prediction)