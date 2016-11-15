# pyDAAL Tutorials
This is a set of tutorials for uisng pyDAAL, i.e. the Python API of [Intel Data Analytics Acceleration Library](https://software.intel.com/en-us/intel-daal). 
It is designed to provide a quick introduction to pyDAAL features and the API
for Python developers who are already familiar with basic concepts and techniques in
machine learning. 

The tutorials are spread across a collection of [Jupyter notebooks](http://jupyter.org/). 
The proper way of using these notebooks is to install [Intel Distribution for
Python](https://software.intel.com/en-us/intel-distribution-for-python) on your
computer, which
consists of a large set of commonly used mathematic and statistical Python
packages that are optimized for Intel architectures. 

### Install Intel Distribution for Python through [conda](http://conda.pydata.org/miniconda.html)
1. Install the latest version of [Miniconda](http://conda.pydata.org/miniconda.html).
    - Choose the Python 3.5 version
2. From the shell prompt (on Windows, use **Anaconda Prompt**), execute these
   commands:

    ```bash
    conda config --add channels intel
    conda create --name idp intelpython3_full python=3
    source activate idp (on Linux and OS X)
    activate idp (on Windows)
    ```
More detailed instructions can be found from [this online article](https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda).

### Notebooks
* [Data management in pyDAAL](./NumericTables_example.ipynb)
* [K-Means and PCA](./kmeans_example.ipynb)
* [Linear regression](./LR_example.ipynb)
* [SVM and multi-class classifier](./SVM_example.ipynb)

Data files used in the tutorials are in the `./mldata` folder. These data files
are downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets).

