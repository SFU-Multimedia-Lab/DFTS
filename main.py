import argparse
import numpy as np
import simmods
from utils import errorCalc
from BrokenModel import BrokenModel as BM
import sys

#modify UI to accept configuration file
#add a packet loss probability as well

def argumentReader():
    '''
        * Read in command line arguments
            ** m: model such as those given at https://keras.io/applications/
            ** l: layer you want to split your model at
            ** i: path to the image
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", help="name of the model", required=True)
    ap.add_argument("-l", "--layer", help="layer you want to split at", required=True)
    ap.add_argument("-i", "--image", help="path to the image", required=True)
    args = vars(ap.parse_args())
    return args

def runSimulation():
    '''
        * Part of the model runs on the device: deviceSim
        * Compression: compress
        * Channel: transmit
        * Other Part runs in the cloud: remoteSim
    '''
    args = argumentReader()

    modelDict = {'xception':'Xception', 'vgg16':'VGG16', 'VGG19':'VGG19', 'resnet50':'ResNet50',
                 'inceptionv3':'InceptionV3', 'inceptionresnetv2':'InceptionResnetV2',
                 'mobilenet':'MobileNet', 'densenet':'DenseNet','nasnet':'NASNet'}

    try:
        model = modelDict[args['model'].lower()]
    except KeyError:
        print("We currently do not support that model. Please try a valid one.")
        sys.exit(1)

    testModel = BM(model, args['layer'])
    testModel.splitModel()
    # print(testModel.deviceModel.summary())
    deviceOut            = simmods.deviceSim(testModel.deviceModel, args['image'], args['model'])
    compressOut          = simmods.compress(deviceOut)
    channelOut, pLen, bS = simmods.transmit(compressOut, 0.96) #second param is the packet loss prob
    # print(channelOut[2000])
    # print(testModel.remoteModel.summary())
    remoteOut            = simmods.remoteSim(testModel.remoteModel, channelOut, pLen, bS)

    exec("from keras.applications."+model.lower()+" import decode_predictions", globals())
    print(decode_predictions(remoteOut, top=5)[0])
    # errorCalc(remoteOut, actualOut)

if __name__ == '__main__':
    runSimulation()
