from utils import preprocess, randomChoice
import numpy as np
from PacketModel import Packet
import time

def deviceSim(model, testImagePath, modelName):
    '''
        * simulate a user device
        * runs the image through the pretrained model up until the specified layer number
    '''
    start_time = time.time()
    preprocessedImageList = preprocess(testImagePath, modelName)
    deviceOut         = model.predict(preprocessedImageList)
    total_time = time.time() - start_time
    print(f"Device cimulation Complete in {total_time}!!")
    return deviceOut

def compress(deviceOut):
    #initially identity function
    return deviceOut

def transmit(compressOut, lossProb, rowsPerPacket):
    #default packet length is one row of the feature map
    #put this in a different function
    start_time = time.time()
    pckts        = Packet(compressOut, rowsPerPacket)
    # print(compressOut[2000])
    lossMatrix   = randomChoice(lossProb, pckts.packetSeq.shape[0])
    # print(lossMatrix)
    # pckts.packetSeq = (pckts.packetSeq)*lossMatrix[:, np.newaxis]
    pckts.packetSeq = [pckts.packetSeq[i]*lossMatrix[i] for i in range(pckts.packetSeq.shape[0])]
    total_time = time.time() - start_time
    print(f"Transmission Complete in {total_time}!!")
    return pckts

def remoteSim(remoteModel ,channelOut):
    data = channelOut.packetToData()
    x = np.reshape(data, (channelOut.bS, channelOut.cols, channelOut.cols, channelOut.kernels))
    return remoteModel.predict(x)
