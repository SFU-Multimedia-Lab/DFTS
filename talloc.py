from modelTasks.tasks import *

def taskAllocater(task, testDir, reshapeDims, batch_size=64, normalize=False):
    """Chooses the task based on the user's options

    # Arguments
        task: integer value denoting the task
        testDir: directory location of the images
        batch_size: number of images to be forwarded through the model at once
        reshapeDims: list containing the dimensions of the reshaped image
        normalize: bool value indicating whether images must be normalized        
    """
    if task==0:
        return CFTask(testDir, reshapeDims, batch_size, normalize)
    if task==1:
        return ODTask(testDir)
