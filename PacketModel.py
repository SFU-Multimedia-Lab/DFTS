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
        self.packetSeq     = self.dataToPacket(data, rowsPerPacket)

    def dataToPacket(self, data, rowsPerPacket):
        start_time = time.time()
        data = data.flatten()
        i = 0
        # print(data.shape[0])
        stepSize = rowsPerPacket*(self.cols)
        data = np.split(data, np.arange(0, data.shape[0], stepSize))
        np.delete(data, 0)
        total_time = time.time() - start_time
        print(f"Packetization complete in {total_time}!!")
        return np.array(data)

    def packetToData(self):
        start_time = time.time()
        pd = np.concatenate(self.packetSeq)
        total_time = time.time() - start_time
        print(f"Converted Packets back to data in {total_time}")
        return pd
