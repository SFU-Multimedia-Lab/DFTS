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
        data = data.flatten()
        i = 0
        print(data.shape[0])
        stepSize = rowsPerPacket*(self.cols)
        data = np.split(data, [i for i in np.arange(0, data.shape[0], stepSize)])
        np.delete(data, 0)
        print("Packetization complete!!")
        return np.array(data)

    def packetToData(self):
        pd = np.concatenate(self.packetSeq)
        print("Converted Packets back to data")
        return pd
