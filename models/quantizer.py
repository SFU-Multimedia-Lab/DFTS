import numpy as np
from utils import closestPowerOfTwo as cpt
import time

class QLayer(object):
    """docstring for QLayer."""
    def __init__(self, nBits):
        super(QLayer, self).__init__()
        self.nBits = nBits

    def bitQuantizer(self, data):
        start_time = time.time()
        self.max = np.max(data)
        self.min = np.min(data)
        np.seterr(divide='ignore', invalid='ignore')

        #refer to deep feature compression for formulae
        # self.typeSize = 'uint'+str(cpt(self.nBits))
        self.quanData = np.round(((data-self.min)/(self.max-self.min))*((2**self.nBits)-1))#.astype(self.typeSize)
        total_time = time.time() - start_time
        print(f"bit quantizer complete in {total_time}!!")

    def inverseQuantizer(self):
        self.quanData = (self.quanData*(self.max-self.min)/((2**self.nBits)-1)) + self.min
        return self.quanData
