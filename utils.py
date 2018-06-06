from keras.preprocessing import image
from keras import losses, metrics
import keras.backend as K
import numpy as np
import importlib

def errorCalc(remoteOut, classValues):
    # print(remoteOut)
    return np.sum(np.equal(remoteOut, classValues))/classValues.shape[0]
    # y_pred = remoteOut
    # y_true = classValues
    # return metrics.categorical_accuracy(y_true, y_pred)


def preprocess(testImage, model):
    '''
        * convert the input image to the size required by the model
        * currently handles only VGG16, EXTEND LATER
    '''
    a = []
    exec("from keras.applications."+model.lower()+" import preprocess_input", globals())
    for i in testImage:
        img = image.load_img(i, target_size=(224,224))
        x   = image.img_to_array(img)
        x = preprocess_input(x)
        a.append(x)
    # img = image.load_img(testImage, target_size=(224,224))
    # x   = image.img_to_array(img)
    # x   = np.expand_dims(x, axis=0)
    # exec("from keras.applications."+model.lower()+" import preprocess_input", globals())
    # x = preprocess_input(x)
    return np.array(a)

def randomChoice(lossProb, lossSize):
    #more robust implementation refer to my TSP implementation
    lossMatrix = np.random.random(lossSize)
    probMatrix = np.full(lossSize, lossProb)
    # probMatrix = np.empty(lossSize)
    # probMatrix.fill(lossProb)
    return np.less_equal(lossMatrix, probMatrix).astype('float64')

def oneHot(classValues, totalClasses):
    y_true = [[0]*totalClasses]*len(classValues)
    for i in range(len(classValues)):
        y_true[i][classValues[i]] = 1
    return np.array(y_true)
