import sys
sys.path.append('..')

import numpy as np

from .utils import *

from models.BrokenModel import BrokenModel as BM
from .simmods import *

from calloc import loadChannel, quantInit

def runSimulation(model, epochs, splitLayer, task, modelDict, transDict):
    task.gatherData()
    dataGen = task.dataFlow()

    model = modelLoader(model, modelDict)

    testModel = BM(model, splitLayer)

    # @timing
    testModel.splitModel()

    print(transDict)
    rowsPerPacket = transDict['rowsperpacket']
    quantization  = transDict['quantization']
    channel       = transDict['channel']
    lossConceal   = transDict['concealment']

    print(rowsPerPacket)
    print(quantization)
    print(list(channel.keys()))
    print(lossConceal)

    channel = loadChannel(channel)
    quant   = quantInit(quantization)

    # for i in range(epochs):
    #     while not dataGen.runThrough:
    #         label, data = dataGen.getNextBatch()
    #         deviceOut = deviceSim(testModel.deviceModel, data)
    #     dataGen.runThrough = False
