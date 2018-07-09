import numpy as np
from PIL import Image

class DataGenerator(object):
    """docstring for DataGenerator."""
    def __init__(self, testData, **params):
        super(DataGenerator, self).__init__()
        self.testData    = testData
        self.reshapeDims = params['reshapeDims']
        self.batch_size  = params['batch_size']
        self.normalize   = params['normalize']
        self.n_classes   = params['n_classes']
        self.batch_index = 0

    def getNextBatch(self):
        currentTestData  = self.preprocess(self.testData[index:index+self.batch_size])
        self.index      += self.batch_size
        return currentTestData

    def preprocess(self, pdata):
        pdata = [pTasks(i) for i in pdata]
        return pdata
