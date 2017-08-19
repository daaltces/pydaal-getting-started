import numpy as np
from daal.data_management import HomogenNumericTable, BlockDescriptor_Float64, readOnly

def getArrayFromNT(table, nrows=0):
    bd = BlockDescriptor_Float64()
    if nrows == 0:
        nrows = table.getNumberOfRows()
    table.getBlockOfRows(0, nrows, readOnly, bd)
    npa = np.copy(bd.getArray())
    table.releaseBlockOfRows(bd)
    return npa

def printNT(table, nrows = 0, message=''):
    npa = getArrayFromNT(table, nrows)
    print(message, '\n', npa)

