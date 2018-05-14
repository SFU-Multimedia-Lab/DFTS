import keras

class BrokenModel(object):
    """docstring for BrokenModel."""
    def __init__(self, modelName, splitLayer):
        super(BrokenModel, self).__init__()
        self.modelName  = modelName
        self.model      = getattr(keras.applications,f"{modelName}")(weights='imagenet', include_top=False)
        self.layers     = [i.name for i in self.model.layers]
        self.splitLayer = splitLayer
        print(self.layers)
        print(self.model.summary())
