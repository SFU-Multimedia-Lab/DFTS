import sys
sys.path.append('..')

import numpy as np

from .utils import *

from models.BrokenModel import BrokenModel as BM
from .simmods import *

def runSimulation(model, epochs, splitLayer, task, modelDict, transDict):
    task.gatherData()
    dataGen = task.dataFlow()

    model = modelLoader(model, modelDict)

    testModel = BM(model, splitLayer)

    # @timing
    testModel.splitModel()

    for i in range(epochs):
        while not dataGen.runThrough:
            label, data = dataGen.getNextBatch()
            deviceOut = deviceSim(testModel.deviceModel, data)
            print(deviceOut.shape)
        dataGen.runThrough = False
