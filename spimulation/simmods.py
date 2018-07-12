import sys
sys.path.append('..')

import numpy as np
import time

from models.packetModel import PacketModel as PM


def deviceSim(model, data):
    '''
        * simulate a user device
        * runs the image through the pretrained model up until the specified layer number
    '''
    start_time = time.time()
    deviceOut         = model.predict(data)
    total_time = time.time() - start_time
    print(f"Device simulation Complete in {total_time}!!")
    return deviceOut

# def compress(deviceOut):
#     #initially identity function
#     return deviceOut
#
def transmit(compressOut, channel, rowsPerPacket):
    start_time   = time.time()
    pckts        = PM(compressOut, rowsPerPacket)

    lossMatrix = channel.simulate(pckts.packetSeq.shape[0]*pckts.packetSeq.shape[1]*pckts.packetSeq.shape[-1])
    lossMatrix = lossMatrix.reshape(pckts.packetSeq.shape[0], pckts.packetSeq.shape[1], pckts.packetSeq.shape[-1])

    receivedIndices = np.where(lossMatrix==1)
    receivedIndices = np.dstack((receivedIndices[0], receivedIndices[1], receivedIndices[2]))

    lostIndices = np.where(lossMatrix==0)
    lostIndices = np.dstack((lostIndices[0], lostIndices[1], lostIndices[2]))

    pckts.packetSeq[lostIndices[:,:,0], lostIndices[:,:,1], :, :, lostIndices[:,:,-1]] = 0

    total_time = time.time() - start_time
    print(f"Transmission Complete in {total_time}!!")
    return (pckts, lossMatrix, receivedIndices, lostIndices)

def errorConceal(interpPackets, pBuffer, receivedIndices, lostIndices, rowsPerPacket):
    return interpPackets(pBuffer, receivedIndices, lostIndices, rowsPerPacket)

def remoteSim(remoteModel ,channelOut):
    data = channelOut.packetToData()
    return remoteModel.predict(data)
