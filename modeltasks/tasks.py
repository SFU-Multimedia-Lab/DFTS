import os
from utils import absoluteFilePaths
from dataGen import DataGenerator as DG

class CFTask(object):
    """docstring for CFTask."""
    def __init__(self, testDir, reshapeDims, batch_size=64, normalize=False):
        super(CFTask, self).__init__()
        self.testDir     = testDir
        self.batch_size  = batch_size
        self.reshapeDims = reshapeDims
        self.normalize   = normalize
        self.gatherData()

    def gatherData(self):
        dirList = os.listdir(self.testDir)
        self.numClasses = len(dirList)
        data    = [absoluteFilePaths(i) for i in dirList]

        classImgArr = []

        for i in range(len(images)):
            exp = [list((int(dirList[i]), images[i][j])) for j in range(len(images[i]))]
            classImgArr.append(exp)

        classImgArr = np.array([item for sublist in classImgArr for item in sublist])
        labels = classImgArr[:, 0].astype(np.int32)
        # labelsTest = np.array([i.split('\\')[-2] for i in classImgArr[:, 1]])
        classImgArr[:, 0] = labels
        self.testData = classImgArr

    def dataFlow(self):
        params = {
            'reshapeDims': self.reshapeDims,
            'batch_size' : self.batch_size,
            'n_classes'  : self.numClasses,
            'normalize'  : self.normalize
        }
        return DG(self.testData, **params)
