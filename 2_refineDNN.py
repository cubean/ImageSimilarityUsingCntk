# -*- coding: utf-8 -*-
from helpers import *
from helpers_cntk import *
locals().update(importlib.import_module("PARAMETERS").__dict__)


################################################
# MAIN
################################################
makeDirectory(workingDir)

import datetime
current_time1 = datetime.datetime.now()

# Load data
lutLabel2Id = loadFromPickle(lutLabel2IdPath)
imgFilenamesTest  = loadFromPickle(imgFilenamesTestPath)
imgFilenamesTrain = loadFromPickle(imgFilenamesTrainPath)

# Generate cntk test and train data, i.e. (image, label) pairs and write
# them to disk since in-memory passing is currently not supported by cntk
dataTest  = getImgLabelMap(imgFilenamesTest,  imgDir, lutLabel2Id)
dataTrain = getImgLabelMap(imgFilenamesTrain, imgDir, lutLabel2Id)
if rf_boBalanceTrainingSet:
    dataTrain = balanceDatasetUsingDuplicates(dataTrain)
writeTable(cntkTrainMapPath, dataTrain)
writeTable(cntkTestMapPath, dataTest)

# Train model
printDeviceType()
print("rf_modelFilename is %s, rf_modelOutputDimension = %s, rf_inputResoluton_width = %s, rf_inputResoluton_height = %s, rf_dropoutRate = %s, rf_l2RegWeight = %s" % 
    (rf_modelFilename, rf_modelOutputDimension, rf_inputResoluton_width, rf_inputResoluton_height, rf_dropoutRate, rf_l2RegWeight))
model = train_model(cntkPretrainedModelPath, cntkTrainMapPath, cntkTestMapPath, rf_inputResoluton_width, rf_inputResoluton_height,
                    rf_maxEpochs, rf_mbSize, rf_maxTrainImages, rf_lrPerMb, rf_momentumPerMb, rf_l2RegWeight,
                    rf_dropoutRate, rf_boFreezeWeights)
model.save(cntkRefinedModelPath)
print("Stored trained model at %s" % cntkRefinedModelPath)

current_time2 = datetime.datetime.now()
print ("This step took time: %s" % (current_time2 - current_time1))

print("DONE. Showing DNN accuracy vs training epoch plot.")
plt.show() # Accuracy vs training epochs plt