import sys
sys.path.append('..')

import numpy as np

from .utils import *

from models.BrokenModel import BrokenModel as BM

def runSimulation(model, epochs, splitLayer, task, modelDict, transDict):
    task.gatherData()
    dataGen = task.dataFlow()

    model = modelLoader(model, modelDict)

    testModel = BM(model, splitLayer)

    # @timing
    testModel.splitModel()

    # print(testModel.deviceModel.summary())
    # print(testModel.remoteModel.summary())

    for i in range(epochs):
        while not dataGen.runThrough:
            label, data = dataGen.getNextBatch()
            print(label.shape)
            print(data.shape)
        dataGen.runThrough = False
