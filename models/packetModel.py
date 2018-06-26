import numpy as np
import time

class PacketModel(object):
    """docstring for PacketModel."""
    def __init__(self, data, rowsPerPacket):
        super(PacketModel, self).__init__()
        self.rowsPerPacket = rowsPerPacket
        self.dataShape     = data.shape
        self.packetSeq     = self.dataToPacket(data)

    def dataToPacket(self, data):
        self.numZeros = 0

        if self.dataShape[1]%self.rowsPerPacket ==0:
            data = np.reshape(data, (self.dataShape[0], -1, self.rowsPerPacket, self.dataShape[2], self.dataShape[3]))
            return data

        self.numZeros = self.rowsPerPacket - (self.dataShape[1]%self.rowsPerPacket)
        zeros         = np.zeros((self.dataShape[0], self.numZeros, self.dataShape[2], self.dataShape[3]))
        data          = np.concatenate((data, zeros), axis=1)
        data          = np.reshape(data, (self.dataShape[0], -1, self.rowsPerPacket, self.dataShape[2], self.dataShape[3]))
        return data

    def packetToData(self):
        if self.numZeros == 0:
            self.packetSeq = np.reshape(self.packetSeq, self.dataShape)
            return self.packetSeq

        self.packetSeq = np.reshape(self.packetSeq, (self.dataShape[0], -1, self.dataShape[2], self.dataShape[3]))
        index          = -1*self.numZeros
        self.packetSeq = self.packetSeq[:, :index, :, :]
        return self.packetSeq
