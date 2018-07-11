from modelTasks.tasks import *

def taskAllocater(task, testDir, reshapeDims, batch_size=64, normalize=False):
    # task is an integer representing the model's application
    #each task is an object having the following attributes:
        #data generator: different according to the structure of the testInput dir
        #data preprocessing
    #0: classification
    #1: object detection

    if task==0:
        return CFTask(testDir, reshapeDims, batch_size, normalize)
    if task==1:
        return ODTask(testDir)
