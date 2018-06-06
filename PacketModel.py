import numpy as np
import time

class Packet(object):
    """docstring for Packet."""
    def __init__(self, data, rowsPerPacket):
        super(Packet, self).__init__()
        self.bS            = data.shape[0]
        self.cols          = data.shape[2]
        self.kernels       = data.shape[-1]
        self.rowsPerPacket = rowsPerPacket
        self.packetSeq, self.numZeros     = self.dataToPacket(data, rowsPerPacket)

    def dataToPacket(self, data, rowsPerPacket):
        numZeros = 0
        start_time = time.time()
        stepSize = rowsPerPacket*(self.cols)
        if data.size%stepSize == 0:
            data = data.ravel()
            data = data.reshape(-1, stepSize)
            self.numZeros = 0
        else:
            s = stepSize-(data.size%stepSize)
            data = data.ravel()
            data = np.append(data, np.zeros(s))
            data = data.reshape(-1, stepSize)
            numZeros = s
        total_time = time.time() - start_time
        print(f"Packetization complete in {total_time}!!")
        return (data, numZeros)

    def packetToData(self):
        start_time = time.time()
        pd = self.packetSeq.ravel()
        total_time = time.time() - start_time
        print(f"Converted Packets back to data in {total_time}")
        return pd
