import sys
sys.path.append('..')

from .utils import *

from models.BrokenModel import BrokenModel as BM

def runSimulation(model, splitLayer, task, modelDict, transDict):
    task.gatherData()
    dataGen = task.dataFlow()

    model = modelLoader(model, modelDict)

    testModel = BM(model, splitLayer)

    # @timing
    testModel.splitModel()
    print(testModel.deviceModel.summary())
    print(testModel.remoteModel.summary())
