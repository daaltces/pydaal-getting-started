import numpy as np
import warnings
from daal.data_management import  (HomogenNumericTable,convertToHomogen_Float64,readOnly,BlockDescriptor_Float64, MergedNumericTable, \
                                   FileDataSource, DataSource, DataSourceIface, InputDataArchive, OutputDataArchive, \
                                   Compressor_Zlib, Decompressor_Zlib, level9, DecompressionStream, CompressionStream)

'''
Arguments: Numeric table, *args = 'head' or 'tail'
Returns array of numeric table. 
If *args is 'head' returns top 5 rows
If *args is 'tail' returns last 5 rows
'''
def getArrayFromNT(nT,*args):
    doubleBlock = BlockDescriptor_Float64()
    if not args:
        firstRow = 0
        lastRow = nT.getNumberOfRows()
        firstCol = 0
        lastCol = nT.getNumberOfColumns()
    else:
        if args[0] == "head":
            firstRow = 0
            lastRow = 5
            firstCol = 0
            lastCol = 5
        if args[0] == "tail":
            firstRow = nT.getNumberOfRows() - 5
            lastRow = nT.getNumberOfRows()
            firstCol = 0
            lastCol = 5
    nT.getBlockOfRows(firstRow, lastRow, readOnly, doubleBlock)
    getArray = doubleBlock.getArray()
    return getArray
'''
Arguments: Numeric table, *args = [intLowerBound, intupperBound]
Returns block of Numeric table within *args column range 
'''

def getBlockOfCols(nT,*args):
    mnT = MergedNumericTable()
    for idx in range(args[0],args[1]):
        doubleBlock = BlockDescriptor_Float64()
        nT.getBlockOfColumnValues(idx, 0, nT.getNumberOfRows(), readOnly, doubleBlock)
        mnT.releaseBlockOfColumnValues(doubleBlock)
    return mnT

'''
Method 1:
    Arguments: Numeric table
    Returns block of Numeric table having all rows and columns of input Numeric Table. 
Method 2:Arguments: Numeric table, Rows = int, Columns = int 
    Returns block of Numeric table within integer values passed as Rows and Column arguments
Method 3: Numeric Table, Rows=[intLowerBound, intUpperBound], Columns = [intLowerBound, intUpperBound]
    Returns block of Numeric table along row and column directions using intLowerBound and intUpperBound passed as parameters in list
Method 4: Numeric table , Rows=[intLowerBound,], Columns = [intLowerBound,]
    Returns block of Numeric table from intLowerBound index through last index
'''
def getBlockOfNumericTable(nT,Rows = 'All', Columns = 'All'):
    from daal.data_management import HomogenNumericTable_Float64, \
    MergedNumericTable, readOnly, BlockDescriptor
    import numpy as np

    # Get First and Last Row indexes
    lastRow = nT.getNumberOfRows()
    if type(Rows)!= str:
        if type(Rows) == list:
            firstRow = Rows[0]
            if len(Rows) == 2: lastRow = min(Rows[1], lastRow)
        else:firstRow = 0; lastRow = Rows
    elif Rows== 'All':firstRow = 0
    else:
        warnings.warn('Type error in "Rows" arguments, Can be only int/list type')
        raise SystemExit

    # Get First and Last Column indexes
    nEndDim = nT.getNumberOfColumns()
    if type(Columns)!= str:
        if type(Columns) == list:
            nStartDim = Columns[0]
            if len(Columns) == 2: nEndDim = min(Columns[1], nEndDim)
        else: nStartDim = 0; nEndDim = Columns
    elif Columns == 'All': nStartDim = 0
    else:
        warnings.warn ('Type error in "Columns" arguments, Can be only int/list type')
        raise SystemExit

    #Retrieve block of Columns Values within First & Last Rows
    #Merge all the retrieved block of Columns Values
    #Return merged numeric table
    mnT = MergedNumericTable()
    for idx in range(nStartDim,nEndDim):
        block = BlockDescriptor()
        nT.getBlockOfColumnValues(idx,firstRow,(lastRow-firstRow),readOnly,block)
        mnT.addNumericTable(HomogenNumericTable_Float64(block.getArray()))
        nT.releaseBlockOfColumnValues(block)
    block = BlockDescriptor()
    mnT.getBlockOfRows (0, mnT.getNumberOfRows (), readOnly, block)
    mnT = HomogenNumericTable (block.getArray ())
    return mnT
'''
Method 1:
    Arguments: csvFileName
    FileDataSource reads all rows from csv file and returns Numeric table
Method 2: 
    Arguments: csvFileName, Rows = int
    FileDataSource reads int number of Rows from csv file and returns Numeric table    
'''

def getNumericTableFromCSV(csvFileName, Rows = 'All'):
    dataSource = FileDataSource (
        csvFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    nT = HomogenNumericTable()
    if type(Rows)!=str: dataSource.loadDataBlock(Rows, nT)
    elif Rows == 'All': dataSource.loadDataBlock(nT)
    else:
        warnings.warn ('Type error in "Rows" arguments, Can be only int')
        raise SystemError
    return nT
		

'''
---------------------------------------------------------------------------------
*************************DATA PORTABILITY HELPER FUNCTION BLOCK STARTS HERE******
---------------------------------------------------------------------------------
'''
'''
call the serialize() function to invoke compress() method
Arguments: serialized numpy array
Returns Compressed numpy array
'''

def compress(arrayData):
    compressor = Compressor_Zlib ()
    compressor.parameter.gzHeader = True
    compressor.parameter.level = level9
    comprStream = CompressionStream (compressor)
    comprStream.push_back (arrayData)
    compressedData = np.empty (comprStream.getCompressedDataSize (), dtype=np.uint8)
    comprStream.copyCompressedArray (compressedData)
    return compressedData

'''
call the deserialize() function to invoke decompress() method
Arguments: deserialized numpy array
Returns decompressed numpy array
'''
def decompress(arrayData):
    decompressor = Decompressor_Zlib ()
    decompressor.parameter.gzHeader = True
    # Create a stream for decompression
    deComprStream = DecompressionStream (decompressor)
    # Write the compressed data to the decompression stream and decompress it
    deComprStream.push_back (arrayData)
    # Allocate memory to store the decompressed data
    bufferArray = np.empty (deComprStream.getDecompressedDataSize (), dtype=np.uint8)
    # Store the decompressed data
    deComprStream.copyDecompressedArray (bufferArray)
    return bufferArray

#-------------------
#***Serialization***
#-------------------
'''
Method 1:
    Arguments: data(type nT/model)
    Returns  dictionary with serailized array (type object) and object Information (type string)
Method 2:
    Arguments: data(type nT/model), fileName(.npy file to save serialized array to disk)
    Saves serialized numpy array as "fileName" argument
    Saves object information as "filename.txt"
 Method 3:
    Arguments: data(type nT/model), useCompression = True
    Returns  dictionary with compressed array (type object) and object information (type string)
Method 4:
    Arguments: data(type nT/model), fileName(.npy file to save serialized array to disk), useCompression = True
    Saves compresseed numpy array as "fileName" argument
    Saves object information as "filename.txt"
'''

def serialize(data, fileName=None, useCompression= False):
    buffArrObjName = (str(type(data)).split()[1].split('>')[0]+"()").replace("'",'')
    dataArch = InputDataArchive()
    data.serialize (dataArch)
    length = dataArch.getSizeOfArchive()
    bufferArray = np.zeros(length, dtype=np.ubyte)
    dataArch.copyArchiveToArray(bufferArray)
    if useCompression == True:
        if fileName != None:
            if len (fileName.rsplit (".", 1)) == 2:
                fileName = fileName.rsplit (".", 1)[0]
            compressedData = compress(bufferArray)
            np.save (fileName, compressedData)
        else:
            comBufferArray = compress (bufferArray)
            serialObjectDict = {"Array Object":comBufferArray,
                                "Object Information": buffArrObjName}
            return serialObjectDict
    else:
        if fileName != None:
            if len (fileName.rsplit (".", 1)) == 2:
                fileName = fileName.rsplit (".", 1)[0]
            np.save(fileName, bufferArray)
        else:
            serialObjectDict = {"Array Object": bufferArray,
                                "Object Information": buffArrObjName}
            return serialObjectDict
    infoFile = open (fileName + ".txt", "w")
    infoFile.write (buffArrObjName)
    infoFile.close ()

#---------------------
#***Deserialization***
#---------------------
'''
Returns deserialized/ decompressed numeric table/model
Input can be serialized/ compressed numpy array or serialized/ compressed .npy file saved to disk
'''
def deserialize(serialObjectDict = None, fileName=None,useCompression = False):
    import daal
    if fileName!=None and serialObjectDict == None:
        bufferArray = np.load(fileName)
        buffArrObjName = open(fileName.rsplit (".", 1)[0]+".txt","r").read()
    elif  fileName == None and any(serialObjectDict):
        bufferArray = serialObjectDict["Array Object"]
        buffArrObjName = serialObjectDict["Object Information"]
    else:
         warnings.warn ('Expecting "bufferArray" or "fileName" argument, NOT both')
         raise SystemExit
    if useCompression == True:
        bufferArray = decompress(bufferArray)
    dataArch = OutputDataArchive (bufferArray)
    try:
        deSerialObj = eval(buffArrObjName)
    except AttributeError :
        deSerialObj = HomogenNumericTable()
    deSerialObj.deserialize(dataArch)
    return deSerialObj
'''
---------------------------------------------------------------------------------
*************************DATA PORTABILITY HELPER FUNCTION ENDS HERE**************
---------------------------------------------------------------------------------
'''





