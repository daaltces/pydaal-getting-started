
This repository consists of various materials introducing PyDAAL (Python API of [Intel Data Analytics Acceleration Library](https://software.intel.com/en-us/intel-daal)) that facilitates Python and Machine Learning practitioners to start off with PyDAAL concepts. 

Additionally, helper functions and classes have been provided to aid frequently performed PyDAAL operations.

# [1-gentle-introductory-series](./1-gentle-introductory-series)

Volume 1, 2 and 3 in PyDAAL Gentle Introduction Series are available as [Jupyter Notebooks](http://jupyter.org/). These volumes are designed to provide a quick introduction to essential features of PyDAAL.
These Jupyter Notebooks offer a collection of code examples that can be executed in the interactive command shell, and helper functions to automate common PyDAAL functionalities.

## How to use?

Install [Intel Distribution for Python](https://software.intel.com/en-us/intel-distribution-for-python) (IDP) through [conda](https://www.continuum.io/downloads). IDP consists of a large set of commonly used mathematical and statistical Python packages that are optimized for Intel architectures. 

1. Install the latest version of [Anaconda](https://www.continuum.io/downloads).    
- Choose the Python 3.5 version2. 

2. From the shell prompt (on Windows, use **Anaconda Prompt**), execute these  commands:

```bash    
    conda create --name idp intelpython3_full python=3 -c intel    
    source activate idp (on Linux and OS X)      
    activate idp (on Windows)    
```
IDP environment is installed with necessary packages and activated to run these notebooks.  
  
More detailed instructions can be found from [this online article](https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda).

# [2-pre-built-helper-classes](./2-pre-built-helper-classes)

Various stages of machine learning model building process are bundled together to constitute one helper function class. These classes are constructed using PyDAAL’s data management and algorithm libraries to achieve a complete model deployment. 

### Stages supported by each helper function classes
1. Training
2. Prediction
3. Model Evaluation and Quality Metrics
4. Trained Model Storage and Portability

More details on all these stages are available in [Volume 3](./1-gentle-introductory-series/volume-3-analytics-model-building-deployment.ipynb).

### Currently, helper function classes are provided for
1. [Linear Regression](./2-helper-function-classes/LinearRegression)
2. [SVM - Binary and Multi-Class classifier](./2-helper-function-classes/SVM)

For practice, usage examples with sample datasets are also provided that utilize these helper function classes.

# [3-custom-modules](./3-custom-modules)

PyDAAL API's have been used to tailor Python modules that support common operations on DAAL's Data Management library.

Import the [customUtils](./3-custom-modules/customUtils) module and explore basic utilities provided for data retrieval and manipulation operations on DAAL's Data Management library

1. getArrayFromNT() : Extracts a numpy array from numeric table
2. getBlockOfNumericTable(): Slices a block of numeric table with specific range of rows and columns
3. getBlockOfCols(): Extracts a block of numeric table within specific range of columns
4. getNumericTableFromCSV(): Reads a CSV file into a numeric table
5. serialize(): Serializes any input data and saves it into a local variable/disk
6. deserialize(): Deserailizes serialized data from a local variable/disk

# [4-interactive-tutorials](./4-interactive-tutorials)

These tutorials are spread across a collection of Jupyter notebooks comprising a theoritical explanation on algorithms and interactive command shell to execute using PyDDAL API.  

### Tutorials Notebooks

* [Data management in pyDAAL](./4-interactive-tutorials/NumericTables_example.ipynb)

* [K-Means and PCA](./4-interactive-tutorials/kmeans_example.ipynb)

* [Linear regression](./4-interactive-tutorials/LR_example.ipynb)

* [SVM and multi-class classifier](./4-interactive-tutorials/SVM_example.ipynb)

* [Online Ridgeregression](./4-interactive-tutorials/Regression_online_example.ipynb)

* [Online Multinomial NaiveBayes](./4-interactive-tutorials/NaiveBayes_online_example.ipynb)

Data files used in the tutorials are in the [mldata](4-interactive-tutorials/) folder. 
These data files are downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets).



