import sys
sys.path.append('..')

import numpy as np

from .utils import *

from models.BrokenModel import BrokenModel as BM
from .simmods import *

from .calloc import loadChannel, quantInit, plcLoader

def runSimulation(model, epochs, splitLayer, task, modelDict, transDict):
    task.gatherData()
    dataGen = task.dataFlow()

    model = modelLoader(model, modelDict)

    testModel = BM(model, splitLayer)

    # @timing
    testModel.splitModel()

    rowsPerPacket = transDict['rowsperpacket']
    quantization  = transDict['quantization']
    channel       = transDict['channel']
    lossConceal   = transDict['concealment']

    channel = loadChannel(channel)
    quant   = quantInit(quantization)
    conceal = plcLoader(lossConceal)

    for i in range(epochs):
        while not dataGen.runThrough:
            label, data = dataGen.getNextBatch()
            deviceOut = deviceSim(testModel.deviceModel, data)

            if quant!='noQuant':
                quant.bitQuantizer(deviceOut)
                deviceOut = quant.quanData
            if channel!='noChannel':
                deviceOut, lossMatrix, receivedIndices, lostIndices = transmit(deviceOut, channel, rowsPerPacket)
                channel.lossMatrix = []
            if conceal!='noConceal':
                deviceOut.packetSeq = errorConceal(conceal, deviceOut.packetSeq, receivedIndices, lostIndices, rowsPerPacket)
            if quant!='noQuant':
                quant.quanData      = deviceOut.packetSeq
                deviceOut.packetSeq = quant.inverseQuantizer()
            remoteOut = remoteSim(testModel.remoteModel, deviceOut)
            loss      = errorCalc(remoteOut, label)
            print(loss)
        dataGen.runThrough = False
