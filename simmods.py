from utils import preprocess
from keras.layers import Input

def deviceSim(model, testImage):
    '''
        * simulate a user device
        * runs the image through the pretrained model up until the specified layer number
    '''
    # print(modelDict)
    # return (2,3)

    preprocessedImage = preprocess(testImage, model)
    deviceOut         = model.predict(preprocessedImage)
    return deviceOut

def compress(deviceOut):
    pass

def transmit():
    pass

def remoteSim(remoteModel ,channelOut):
    pass
