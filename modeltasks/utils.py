from PIL import Image
import os

def absoluteFilePaths(directory):
   dirList = os.listdir(directory)
   dirList = [os.path.join(directory, i) for i in dirList]
   dirList = [os.path.abspath(i) for i in dirList]
   return dirList

def pTasks(image, reshapeDims, norm):
    I = Image.open(image)
    if reshapeDims != (-1, ):
        I = I.resize(reshapeDims)
    I = np.asarray(I)

    if norm:
        I = normalize(I)
    return I

def normalize(image):
    #currently norm accross each channel separately
    image[:, :, 0] -= np.mean(image[:, :, 0])
    image[:, :, 0] /= np.std(image[:, :, 0])

    image[:, :, 1] -= np.mean(image[:, :, 1])
    image[:, :, 1] /= np.std(image[:, :, 1])

    image[:, :, 2] -= np.mean(image[:, :, 2])
    image[:, :, 2] /= np.std(image[:, :, 2])

    return image
