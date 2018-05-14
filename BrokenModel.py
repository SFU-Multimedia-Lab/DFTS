import keras
from keras.layers import Input
from keras.models import Model

class BrokenModel(object):
    """docstring for BrokenModel."""
    def __init__(self, modelName, splitLayer):
        super(BrokenModel, self).__init__()
        self.modelName  = modelName
        self.model      = getattr(keras.applications,f"{modelName}")(weights='imagenet', include_top=False)
        self.layers     = [i.name for i in self.model.layers]
        self.splitLayer = splitLayer
        self.layerLoc   = self.layers.index(self.splitLayer)
        # print(self.layers)
        # print(self.model.summary())

    def splitModel(self):
        self.deviceModel = Model(inputs=self.model.input, outputs=self.model.layers[self.layerLoc].output)
        print(self.deviceModel.summary())
        rmInput          = Input(self.model.layers[self.layerLoc+1].input_shape[1:])
        self.remoteModel = rmInput
        for layer in self.model.layers[self.layerLoc+1:]:
            self.remoteModel = layer(self.remoteModel)
        self.remoteModel = Model(inputs=rmInput, outputs=self.remoteModel)
        print(self.remoteModel.summary())