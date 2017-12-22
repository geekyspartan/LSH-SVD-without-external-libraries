########################################
## assignment 2 - Stony Brook University
## Fall 2017

from pyspark import SparkContext
import sys
import io
import numpy as np
import zipfile
from tifffile import TiffFile
import hashlib
from numpy.linalg import svd

#get the mean, std and vH from random sample and use it to do SVD on all the other images
#Using V on each image to do PCA
def calculateSvdFromVH(sddPartition, vMatrix, mu, std):
    for i in range(0, len(std)):
        if std[i] == 0:
            std[i] = 1
    std_svdMatrix = (sddPartition - mu) / std
    return np.matmul(std_svdMatrix, vMatrix)

#this is used for getting vH, std, mu for a random sample. This will be used later for doing PCA on all the other images.
def imagesSvd(sddPartition):
    svdMatrix = []
    fileNames = []
    count = 0
    for x in sddPartition:
        if count > 10:
            break

        count += 1
        fileNames.append(x[0])
        svdMatrix.append(x[1][0])

    # in order for SVD to perform PCA, the columns must be standardized:
    # (also known as “zscore”: mean centers and divides by stdev)
    mu, std = np.mean(svdMatrix, axis=0), np.std(svdMatrix, axis=0)
    for i in range(0, len(std)):
        if std[i] == 0:
            std[i] = 1
    std_svdMatrix = (svdMatrix - mu) / std

    U, s, Vh = svd(std_svdMatrix, full_matrices=0)

    # run singular value decomposition.
    #https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.svd.html
    # Read more about svd options from above link
    #below code is used to get 10*4900 vH matrix
    Vh = Vh.T
    Vh = Vh[:, 0:10]
    return (Vh, std, mu)

#removes hash keys and check whether needed file exists in a particular hashBucket or not
def groupSimilarFiles(files, fileNames):
    outputFiles = fileNames
    
    res = list()
    for file in outputFiles:
        if file in files:
            x = list(files)
            x.remove(file)
            res.append((file, x),)

    return res


def lsh(feature_vector):
    #will create one key using band_id(1.8) + bucket_id
    #total # of different hashes for a given feature vector will be equal to bands
    bands = 8
    bandNumber = 1
    signature = feature_vector[1]
    rows = 16
    hash_brackets = {}
    for i in range(0, 128, rows):
        key = signature[i:i+rows]
        sum = 0
        keyString = ""
        for y in key:
            keyString += str(y)
    
        # taking 120*16 row buckets gives optimum output
        # adding 1, because I want to start my buckets from 1 not from zeroes.
        row_id = (int(keyString, 2) % (rows)) + 1
        final_key = int(str(bandNumber) + str(row_id))
        try:
            hash_brackets[final_key].append(feature_vector[0])
        except KeyError:
            #this won't be called ever.
            hash_brackets[final_key] = feature_vector[0]
        bandNumber += 1
    return hash_brackets.items()

#this is used to create signature, every bucket is of 38 bits and adding the remaining bits in the front one by one till I can.
def createSignature(feature_vector):
    count = 0
    output = []
    extraBits = 4900 - 38*128
    end = 0
    start = 0
    for i in range(0, 128):
        start = end
        if extraBits > 0:
            end = end + 39
            extraBits -= 1
        else:
            end = end + 38
        hash_hex = hashlib.md5(feature_vector[0][start:end]).hexdigest()
        hash_int = int(hash_hex, 16)
        hash_binary = "{0:b}".format(hash_int)
        #picking a random bit, just 3, it can be any bit.
        output.append(int(hash_binary[3]))

    return output

def intensityCalculation(array):
    outputArray = np.zeros(shape=(500,500), dtype=np.float64)
    for i in range(0, 500):
        for j in range(0, 500):
            r,g,b,infrared = array[i][j]
            outputArray[i][j] = (((float(r)*0.33+float(g)*0.33+float(b)*0.33))*(float(infrared)/100.0))
    return reduceResolution(outputArray)

def reduceResolution(input):
    outputArray = np.zeros(shape=(50,50), dtype=np.float64)
    split1 = np.split(input, 50, axis = 0)
    row = 0
    col = 0
    for array in split1:
        split2 = np.split(array, 50, axis = 1)
        col = 0
        for output in split2:
            outputArray[row][col] = np.mean(output)
            col += 1
        row += 1
    return featureVector(outputArray)

def featureVector(input):
    row_diff = np.diff(input, axis=1) #50*49
    col_diff = np.diff(input, axis=0) #49*50
    row_diff = np.where(row_diff > 1, 1, (np.where(row_diff < -1, -1, 0)))
    col_diff = np.where(col_diff > 1, 1, (np.where(col_diff < -1, -1, 0)))
    res = np.hstack((row_diff.reshape(1,2450),col_diff.reshape(1,2450)))
    return res


def getOrthoTif(filename, zfBytes):
    filename = filename[filename.rfind('/') + 1:]
    #given a zipfile as bytes (i.e. from reading from a binary file),
    # return a np array of rgbx values for each pixel
    bytesio = io.BytesIO(zfBytes)
    zfiles = zipfile.ZipFile(bytesio, "r")
    #find tif:
    for fn in zfiles.namelist():
        if fn[-4:] == '.tif':#found it, turn into array:
            tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
            result = []
            tifArray = tif.asarray()
            splitBy = int(tifArray.shape[0]/500)
            split1 = np.split(tifArray, splitBy, axis = 0)
            arrayNumber = 0
            for array in split1:
                split2 = np.split(array, splitBy, axis = 1)
                for output in split2:
                    outputFileName = filename + "-" + str(arrayNumber)
                    arrayNumber = arrayNumber + 1
                    result.append((outputFileName, output))
            return result

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    if len(sys.argv) > 1:
        directoryPath = sys.argv[1]
    else:
        directoryPath = "/Users/anuragarora/Documents/GitHub/LSH-SVD-without-external-libraries/sampleimages/*"

    #unzip the zip file and use it

    outputFileTxt = open("sample_output.txt","w")
    pairRdd = sc.binaryFiles(directoryPath)

    singleRdd = pairRdd.flatMap(lambda file: getOrthoTif(file[0], file[1]))

    featureVectorOfImages = singleRdd.mapValues(lambda x : intensityCalculation(x))

    allFileNames = featureVectorOfImages.keys().map(lambda x: x[x.rfind('/') + 1:]).collect()
    output = featureVectorOfImages.mapValues(lambda x: createSignature(x)).flatMap(lambda x: lsh(x)).groupByKey().mapValues(list)
    output = output.flatMap(lambda x: groupSimilarFiles(x[1], allFileNames)).groupByKey().mapValues(lambda x: list(set(list(set().union(*x)))))
    
    similarFilesDict = dict()
    for item in output.collect():
        similarFilesDict[item[0]] = item[1]

    similarFilesList = {x for v in similarFilesDict.values() for x in v}
    similarFilesList |= set(similarFilesDict.keys())
    listSample = featureVectorOfImages.takeSample(False, 10)
    vh, std, mu = imagesSvd(listSample)
    svdRdd = featureVectorOfImages.mapValues(lambda x: calculateSvdFromVH(x, vh, mu, std)).filter(lambda a: a[0] in similarFilesList)
    lowDimensionImages = svdRdd.collect()

    print("\n\n************    Similarity     *****************\n")

    lowDimensionImagesDict = dict()
    for item in lowDimensionImages:
        lowDimensionImagesDict[item[0]] = item[1]

    distance = dict()
    outputFileTxt.write("\n************    Similarity     *****************\n")
    eDistance = 0
    for reference,images in similarFilesDict.items():
        for img in images:
            for i in range(0,10):
                eDistance += (lowDimensionImagesDict[reference][0][i] - lowDimensionImagesDict[img][0][i])**2
            eDistance =  np.sqrt(eDistance)
            distance["distance between " + reference + " and " + img] = eDistance
            eDistance = 0
        sortedDistance = str(sorted(distance.items(), key=lambda dist: dist[1]))
        outputFileTxt.write(sortedDistance)
        outputFileTxt.write("\n\n*****************************\n")
        print(sortedDistance)
        print("*****************************")
        distance = dict()





