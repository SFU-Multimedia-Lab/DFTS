import os
import keras
import time
import numpy as np
from keras.models import load_model

def modelLoader(model, modelDict, customObjects):
    """Loads the desired keras model from the disk

    # Arguments
        model: desired keras model or path to that model
        modelDict: dictionary containing official keras models

    # Returns
        A keras model, containing its weights and architecture
    """
    if model in modelDict.values():
        model = getattr(keras.applications,f"{model}")()
        return model
    else:
        model = load_model(model, custom_objects=customObjects)
        return model

def timing(func):
    t1  = time.time()
    res = func(*args, **kwargs)
    t2  = time.time()
    print(t1-t2)

def errorCalc(remoteOut, classValues):
    """Calculate teh accuracy of the prediction

    # Arguments
        remoteOut: prediction from the model in the cloud
        classValues: true labels of the data

    # Returns
        Accuracy of predictions, a number between 0 and 1
    """
    predictions = np.argmax(remoteOut, axis=1)
    return np.sum(np.equal(predictions, classValues))/classValues.shape[0]

def createFile(quant, conceal, splitLayer):
    """Creates a data file based on the desired options
    """
    fileName = splitLayer+"_"
    if quant!="noQuant":
        fileName += f"{quant.nBits}BitQuant_"
    if conceal!="noConceal":
        fileName += "EC"
    fileName += ".npy"
    return fileName
