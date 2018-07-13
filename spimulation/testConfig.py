import sys
sys.path.append('..')

import numpy as np
import os

from .utils import *

from models.BrokenModel import BrokenModel as BM
from .simmods import *

from .calloc import loadChannel, quantInit, plcLoader

def runSimulation(model, epochs, splitLayer, task, modelDict, transDict, simDir):
    """Runs a simulation based on the given parameters.

    Forwaards the data through the model on the device, transmits it, forwards it through the model
    on the cloud and then generates predictions.
    """
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

    fileName = createFile(quant, conceal, splitLayer)
    fileName = os.path.join(simDir, fileName)

    testData = []

    for i in range(epochs):
        epochLoss = []
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
            epochLoss.append(loss)
            print(loss)
        epochLoss = np.array(epochLoss)
        testData.append(np.array([i, np.mean(epochLoss)]))
        dataGen.runThrough = False
    np.save(fileName, np.array(testData))
