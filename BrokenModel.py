import keras
from keras.layers import Input
from keras.models import Model
import numpy as np

class BrokenModel(object):
    """docstring for BrokenModel."""
    def __init__(self, modelName, splitLayer):
        super(BrokenModel, self).__init__()
        self.modelName  = modelName
        #modify below function to suit different models
        self.model      = getattr(keras.applications,f"{modelName}")(weights='imagenet', classes=1000)
        self.layers     = [i.name for i in self.model.layers]
        self.splitLayer = splitLayer
        self.layerLoc   = self.layers.index(self.splitLayer)

    def splitModel(self):
        self.deviceModel = Model(inputs=self.model.input, outputs=self.model.layers[self.layerLoc].output)
        rmInput          = Input(self.model.layers[self.layerLoc+1].input_shape[1:])
        self.remoteModel = rmInput
        for layer in self.model.layers[self.layerLoc+1:]:
            self.remoteModel = layer(self.remoteModel)
        self.remoteModel = Model(inputs=rmInput, outputs=self.remoteModel)

        for i in range(1, len(self.remoteModel.layers)):
            self.remoteModel.layers[i].set_weights(self.model.get_layer(self.layers[self.layerLoc+i]).get_weights())
