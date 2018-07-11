import numpy as np

class RLC(object):
    """docstring for RLC."""
    def __init__(self, lossProb):
        super(RLC, self).__init__()
        self.lossProb   = lossProb
        self.lossMatrix = []

    def simulate(self, lossSize):
        self.lossMatrix = np.random.random(lossSize)
        probMatrix      = np.full(lossSize, self.lossProb)

        return np.greater_equal(lossMatrix, probMatrix).astype('float64')
