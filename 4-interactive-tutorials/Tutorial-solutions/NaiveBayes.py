for i in range(nBlocks):
    # Load new block of data from CSV file
    xTrain = createSparseTable('./mldata/20newsgroups.coo.' + str(i + 1) + '.csv', nFeatures)
    # Load new block of labels from CSV file
    labelsDataSource = FileDataSource(
        './mldata/20newsgroups.labels.' + str(i + 1) + '.csv',
        DataSourceIface.doAllocateNumericTable, DataSourceIface.doDictionaryFromContext
    )
    labelsDataSource.loadDataBlock()
    yTrain = labelsDataSource.getNumericTable()


    # Set input
    #
    # YOUR CODE HERE
    #
    # There are two pieces of input to be set: data and labels. You should
    # use the 'input.set' member methods of the nbTrain algorithm object.
    # The input IDs to use are 'classifier.training.data' and 'classifier.training.labels'
    # respectively.
    nbTrain.input.set(classifier.training.data,xTrain)
    nbTrain.input.set(classifier.training.labels,yTrain)
    
    # Compute
    #
    # YOUR CODE HERE
    #
    # Call the 'compute()' method of your algorithm object to update the partial model.
    nbTrain.compute()

model = nbTrain.finalizeCompute().get(classifier.training.model)