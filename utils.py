from keras.preprocessing import image
import numpy as np
import importlib

def errorCalc(remoteOut, actualOut):
    pass

def preprocess(testImage, model):
    '''
        * convert the input image to the size required by the model
        * currently handles only VGG16, EXTEND LATER
    '''
    img = image.load_img(testImage, target_size=(224,224))
    x   = image.img_to_array(img)
    x   = np.expand_dims(x, axis=0)
    exec("from keras.applications."+model.lower()+" import preprocess_input", globals())
    x = preprocess_input(x)
    return x

def randomChoice(lossProb, lossSize):
    #more robust implementation refer to my TSP implementation
    lossMatrix = np.random.random(lossSize)
    probMatrix = np.empty(lossSize)
    probMatrix.fill(lossProb)
    return np.less_equal(lossMatrix, probMatrix).astype('float32')
