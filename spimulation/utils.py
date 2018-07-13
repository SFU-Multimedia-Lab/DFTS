import os
import keras
import time
import numpy as np
from keras.models import load_model

def modelLoader(model, modelDict):
    if model in modelDict.values():
        model = getattr(keras.applications,f"{model}")()
        return model
    else:
        model = load_model(model)
        return model

def timing(func):
    t1  = time.time()
    res = func(*args, **kwargs)
    t2  = time.time()
    print(t1-t2)

def errorCalc(remoteOut, classValues):
    predictions = np.argmax(remoteOut, axis=1)
    return np.sum(np.equal(predictions, classValues))/classValues.shape[0]

def createFile(quant, conceal, splitLayer):
    fileName = ""
    if quant!="noQuant":
        fileName += f"{quant.nBits}BitQuant_"
    if conceal!="noConceal":
        fileName += "EC"
    fileName += ".npy"
    return fileName
