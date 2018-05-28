import numpy as np

class Packet(object):
    """docstring for Packet."""
    def __init__(self, data, rowsPerPacket):
        super(Packet, self).__init__()
        self.bS            = data.shape[0]
        self.cols          = data.shape[2]
        self.kernels       = data.shape[-1]
        self.rowsPerPacket = rowsPerPacket
        self.packetSeq     = self.dataToPacket(data, rowsPerPacket)

    def dataToPacket(self, data, rowsPerPacket):
        cols = data.shape[2]
        data = data.flatten()
        pseq = []
        #try to produce a cleaner version of the for loop
        i = 0
        while True:
            temp = data[i:rowsPerPacket*cols]
            pseq.append(temp)
            i   += rowsPerPacket*cols
            if i>=data.shape[0]:
                i -= rowsPerPacket*cols
                break
        temp = data[i:-1]
        pseq.append(temp)
        return np.array(pseq)

    def packetToData(self):
        pd = np.array([item for sublist in self.packetSeq for item in self.packetSeq])
        return pd
