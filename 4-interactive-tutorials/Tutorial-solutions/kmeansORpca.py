 def compute(self, data):
        """Compute PCA the input data

        Args:
           data: Input data 
        """
               
        # Create an algorithm object for PCA
        #
        # YOUR CODE HERE
        # The algorithm class you need is either Batch_Float64SvdDense or Batch_Float64CorrelationDense.
        if self.method_ == 'correlation':
            algorithm = pca.Batch_Float64CorrelationDense()
        elif self.method_ == 'svd':
            algorithm = pca.Batch_Float64SvdDense()


        # Set input
        #
        # YOUR CODE HERE
        # Use the 'input.setDataset' member method of the algorithm class to set input.         
        # Th signature of the method is: input.setDataset(InputID, input)
        # You should use 'pca.data' for InputID.
        algorithm.input.setDataset(pca.data,data)
        # compute
        #
        result = algorithm.compute()
        # YOUR CODE HERE
        # You should store the return value of compute to 'result'
        
        
        self.eigenvalues_ = result.get(pca.eigenvalues)
        self.eigenvectors_ = result.get(pca.eigenvectors)