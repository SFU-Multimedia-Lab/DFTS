from utils import preprocess, randomChoice
import numpy as np

def deviceSim(model, testImagePath, modelName):
    '''
        * simulate a user device
        * runs the image through the pretrained model up until the specified layer number
    '''

    preprocessedImageList = preprocess(testImagePath, modelName)
    deviceOut         = model.predict(preprocessedImageList)
    return deviceOut

def compress(deviceOut):
    #initially identity function
    return deviceOut

def transmit(compressOut, lossProb):
    #default packet length is one row of the feature map
    #put this in a different function
    packetLength = compressOut.shape[2]
    batchSize    = compressOut.shape[0]
    compressOut  = compressOut.flatten().reshape(-1, compressOut.shape[1])
    # print(compressOut[2000])
    lossMatrix   = randomChoice(lossProb, compressOut.shape[0])
    # print(lossMatrix)
    return (compressOut*lossMatrix[:, None], packetLength, batchSize)

def remoteSim(remoteModel ,channelOut, pLen, bS):
    x = channelOut.reshape(bS, pLen, pLen, -1)

    #assemble packets: create another function

    return remoteModel.predict(x)
