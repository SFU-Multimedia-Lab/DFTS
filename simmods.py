from utils import preprocess, randomChoice
import numpy as np
from PacketModel import Packet

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

def transmit(compressOut, lossProb, rowsPerPacket):
    #default packet length is one row of the feature map
    #put this in a different function
    pckts        = Packet(compressOut, rowsPerPacket)
    # print(compressOut[2000])
    lossMatrix   = randomChoice(lossProb, pckts.packetSeq.shape[0])
    # print(lossMatrix)
    # pckts.packetSeq = (pckts.packetSeq)*lossMatrix[:, np.newaxis]
    pckts.packetSeq = [pckts.packetSeq[i]*lossMatrix[i] for i in range(pckts.packetSeq.shape[0])]
    print("Transmission Complete!!")
    return pckts

def remoteSim(remoteModel ,channelOut):
    data = channelOut.packetToData()
    x = np.reshape(data, (-1, channelOut.cols, channelOut.cols, channelOut.kernels))

    #assemble packets: create another function
    print("Remote Simulation complete!!")
    return remoteModel.predict(x)
