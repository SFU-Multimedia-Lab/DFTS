import os
import keras
import time
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
