import argparse
import numpy as np
import simmods
from utils import errorCalc

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
    deviceOut, actualOut = simmods.deviceSim(args['model'], args['layer'], args['image'])
    print("hello")
    compressOut          = simmods.compress(deviceOut)
    channelOut           = simmods.transmit()
    remoteOut            = simmods.remoteSim(channelOut)
    errorCalc(remoteOut, actualOut)

if __name__ == '__main__':
    runSimulation()
