from utils import preprocess

def deviceSim(model, testImage, modelName):
    '''
        * simulate a user device
        * runs the image through the pretrained model up until the specified layer number
    '''
    preprocessedImage = preprocess(testImage, modelName)
    deviceOut         = model.predict(preprocessedImage)
    return deviceOut

def compress(deviceOut):
    pass

def transmit():
    pass

def remoteSim(remoteModel ,channelOut):
    pass
