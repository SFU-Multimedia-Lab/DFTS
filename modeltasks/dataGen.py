import numpy as np
from PIL import Image
from .utils import pTasks

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
        self.runThrough  = False

    def getNextBatch(self):
        currentTestData = []
        if self.batch_index>=len(self.testData):
            #assume that batch_size<number of test images
            self.batch_index = 0
            currentTestData  = self.preprocess(self.testData[self.batch_index:self.batch_index+self.batch_size])
        elif self.batch_index + self.batch_size >= len(self.testData) and self.batch_index<len(self.testData):
            currentTestData  = self.preprocess(self.testData[self.batch_index:])
            self.batch_index = len(self.testData)
        else:
            currentTestData  = self.preprocess(self.testData[self.batch_index:self.batch_index+self.batch_size])
        if self.batch_index==len(self.testData):
            self.runThrough = True
        self.batch_index      += self.batch_size
        return currentTestData

    def preprocess(self, pdata):
        labels = []
        data   = []
        for i in pdata:
            l, d = pTasks(i, self.reshapeDims, self.normalize)
            labels.append(l)
            data.append(d)
        return (np.array(labels), np.array(data))
