import argparse
import numpy as np
import simmods
from utils import errorCalc
from BrokenModel import BrokenModel as BM
import sys
import os
import time
import keras.backend as K


#modify UI to accept configuration file
#add a packet loss probability as well

def argumentReader():
    '''
        * Read in command line arguments
            ** m: model such as those given at https://keras.io/applications/
            ** l: layer you want to split your model at
            ** i: path to the image/s
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
    start_time = time.time()
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
    filenames = []
    if os.path.isdir(args['image']):
        filenames = os.listdir(args['image'])
        filenames = [os.path.join(args['image'], i) for i in filenames]
    else:
        filenames.append(args['image'])

    lossList = np.arange(0, 1, 0.05)
    lossData = []
    for l in lossList:
        a = time.time()
        deviceOut            = simmods.deviceSim(testModel.deviceModel, filenames, args['model'])
        compressOut          = simmods.compress(deviceOut)
        channelOut           = simmods.transmit(compressOut, l, 1) #second param is the packet loss prob
        # print(channelOut[2000])
        # print(testModel.remoteModel.summary())
        start_time = time.time()
        remoteOut            = simmods.remoteSim(testModel.remoteModel, channelOut)
        total_time = time.time() - start_time
        print(f"Remote Simulation complete in {total_time}!!")

        start_time = time.time()
        # exec("from keras.applications."+model.lower()+" import decode_predictions", globals())
        # print(decode_predictions(remoteOut, top=1))
        # y_pred = np.argmax(remoteOut, axis=1)
        classValues = np.full(len(filenames), 784)  #push this to utils
        # print(remoteOut[:, [784]])
        loss = np.mean(errorCalc(remoteOut, classValues))
        # accuracy = 1-loss
        # print(f"accuracy = {accuracy}")
        temp = np.array([l, loss])
        lossData.append(temp)
        total_time = time.time() - start_time
        print(loss)
        print(f"Time to calc error{total_time}")
        t = time.time() - a
        print(f"Time for one simulation {t}")
        print("------------------------------")
    lossData = np.array(lossData)
    filename = args['layer'] + '_lossData5.npy'
    np.save(filename, lossData)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    runSimulation()
