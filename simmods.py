from utils import preprocess, randomChoice
import numpy as np
# from PacketModel import Packet
import time
# from plc.linearInterp import interpPackets
from plc.nearestNeighbours import NNInterp as interpPackets
from gbChannel import GBC

from models.packetModel import PacketModel as PM

np.set_printoptions(threshold=np.nan)


def deviceSim(model, testImagePath, modelName):
    '''
        * simulate a user device
        * runs the image through the pretrained model up until the specified layer number
    '''
    start_time = time.time()
    preprocessedImageList = preprocess(testImagePath, modelName)
    deviceOut         = model.predict(preprocessedImageList)
    total_time = time.time() - start_time
    print(f"Device simulation Complete in {total_time}!!")
    return deviceOut

def compress(deviceOut):
    #initially identity function
    return deviceOut

def transmit(compressOut, lossProb, rowsPerPacket, burstLength):
    #default packet length is one row of the feature map
    #put this in a different function

    #random loss channel
    # start_time = time.time()
    # pckts        = PM(compressOut, rowsPerPacket)
    # # print(compressOut[2000])
    # lossMatrix   = randomChoice(lossProb, pckts.packetSeq.shape[0])
    # print(time.time()-start_time)

    #gilbert loss
    start_time = time.time()
    pckts        = PM(compressOut, rowsPerPacket)
    gb = GBC(lossProb, burstLength)

    lossMatrix = gb.simulate(pckts.packetSeq.shape[0]*pckts.packetSeq.shape[1]*pckts.packetSeq.shape[-1])
    lossMatrix = lossMatrix.reshape(pckts.packetSeq.shape[0], pckts.packetSeq.shape[1], pckts.packetSeq.shape[-1])

    receivedIndices = np.where(lossMatrix==1)
    receivedIndices = np.dstack((receivedIndices[0], receivedIndices[1], receivedIndices[2]))

    lostIndices = np.where(lossMatrix==0)
    lostIndices = np.dstack((lostIndices[0], lostIndices[1], lostIndices[2]))

    pckts.packetSeq[lostIndices[:,:,0], lostIndices[:,:,1], :, :, lostIndices[:,:,-1]] = 0

    total_time = time.time() - start_time
    print(f"Transmission Complete in {total_time}!!")
    return (pckts, lossMatrix, receivedIndices, lostIndices)

def errorConceal(pBuffer, receivedIndices, lostIndices, rowsPerPacket, plcKind):
    #will change for tensor completion
    plcMethod = plcKind[0].lower() #will help in importing different plc methods
    kind      = plcKind[1].lower()
    return interpPackets(pBuffer, receivedIndices, lostIndices, rowsPerPacket)


def remoteSim(remoteModel ,channelOut):
    data = channelOut.packetToData()
    return data, remoteModel.predict(data)
