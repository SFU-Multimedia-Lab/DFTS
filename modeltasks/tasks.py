import os
import numpy as np
from .utils import absoluteFilePaths
from .dataGen import DataGenerator as DG

class CFTask(object):
    """Initializes the simulator for classfication.
    """
    def __init__(self, testDir, reshapeDims, batch_size=64, normalize=False):
        """Initialize the class and gather the test data

        # Arguments
            testDir: top level directory containing the images
            batch_size: number of images to be forwarded through the model at once
            reshapeDims: list containing the dimensions of the reshaped image
            normalize: bool value indicating whether images must be normalized
        """
        super(CFTask, self).__init__()
        self.testDir     = testDir
        self.batch_size  = batch_size
        self.reshapeDims = reshapeDims
        self.normalize   = normalize
        self.gatherData()

    def gatherData(self):
        """Organizes the data into an array containing the labels and path to the images.
        """
        dirList = os.listdir(self.testDir)
        self.numClasses = len(dirList)
        dirList = [os.path.join(self.testDir, i) for i in dirList]
        images  = [absoluteFilePaths(i) for i in dirList]

        classImgArr = []

        for i in range(len(images)):
            exp = [list((int(dirList[i].split('\\')[-1]), images[i][j])) for j in range(len(images[i]))]
            classImgArr.append(exp)

        classImgArr = np.array([item for sublist in classImgArr for item in sublist])
        labels = classImgArr[:, 0].astype(np.int32)
        # labelsTest = np.array([i.split('\\')[-2] for i in classImgArr[:, 1]])
        classImgArr[:, 0] = labels
        self.testData = classImgArr

    def dataFlow(self):
        """Create a data generator based on the given parameters

        # Returns
            A Data Generator object.
        """
        params = {
            'reshapeDims': self.reshapeDims,
            'batch_size' : self.batch_size,
            'n_classes'  : self.numClasses,
            'normalize'  : self.normalize
        }
        return DG(self.testData, **params)
