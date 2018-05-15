from utils import preprocess, randomChoice

def deviceSim(model, testImage, modelName):
    '''
        * simulate a user device
        * runs the image through the pretrained model up until the specified layer number
    '''
    preprocessedImage = preprocess(testImage, modelName)
    deviceOut         = model.predict(preprocessedImage)
    return deviceOut

def compress(deviceOut):
    #initially identity function
    return deviceOut

def transmit(compressOut, lossProb):
    #default packet length is one row of the feature map
    compressOut = compressOut.flatten().reshape(-1, compressOut.shape[1])
    print(compressOut[2000])
    lossMatrix  = randomChoice(lossProb, compressOut.shape[0])
    # print(lossMatrix)
    return compressOut*lossMatrix[:, None]

def remoteSim(remoteModel ,channelOut):
    pass
