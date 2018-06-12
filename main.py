import argparse
import numpy as np
import simmods
from utils import errorCalc
from BrokenModel import BrokenModel as BM
import sys
import os
import time
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import keras
from quantizer import QLayer as QL

# warnings.simplefilter(action='ignore', category=FutureWarning)

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
    t1 = time.time()
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
    tot = time.time() - t1
    print(f"Time to split model {tot}")
    # print(testModel.deviceModel.summary())
    filenames = []
    if os.path.isdir(args['image']):
        filenames = os.listdir(args['image'])
        filenames = [os.path.join(args['image'], i) for i in filenames]
    else:
        filenames.append(args['image'])

    lossList = np.arange(0.2, 1.2, 0.2)
    # lossList = [1]
    packetList = np.arange(1, 6)
    num_epochs = 5
    dir = os.path.join('..', 'testData')
    classValues = np.full(len(filenames), 235)  #push this to utils
    for l in lossList:
        for p in packetList:
            lossData = []
            for i in range(num_epochs):
                print(f"Epoch:{i}")
                a = time.time()
                deviceOut            = simmods.deviceSim(testModel.deviceModel, filenames, args['model'])
                compressOut          = simmods.compress(deviceOut)
                channelOut,loss      = simmods.transmit(compressOut, l, p) #second param is the packet loss prob

                nBits = 10
                qnt = QL(nBits)
                qnt.bitQuantizer(channelOut.packetSeq)
                channelOut.packetSeq = qnt.inverseQuantizer()

                '''
                    Multiprocessing to speed up the non gpu implementations
                '''


                start_time = time.time()
                remoteOut            = simmods.remoteSim(testModel.remoteModel, channelOut)
                total_time = time.time() - start_time
                print(f"Remote Simulation complete in {total_time}!!")
                start_time = time.time()
                predictions = np.argmax(remoteOut, axis=1)
                loss = errorCalc(predictions, classValues)
                temp = np.array([i, loss])
                lossData.append(temp)
                total_time = time.time() - start_time
                print(loss)
                print(f"Time to calc error{total_time}")
                t = time.time() - a
                print(f"Time for one simulation {t}")
                print("------------------------------")
            index = np.where(lossList==l)[0][0]
            filename = f"{index}Loss_{nBits}Quantization_"+f"{p}Packet_"+args['layer'] + '.npy'
            lossData = np.array(lossData)
            np.save(os.path.join(dir, filename), lossData)

    # l = 1
    # p = 1
    # a = time.time()
    # deviceOut                     = simmods.deviceSim(testModel.deviceModel, filenames, args['model'])
    # compressOut                   = simmods.compress(deviceOut)
    # channelOut, loss              = simmods.transmit(compressOut, l, p) #second param is the packet loss prob

    # start_time = time.time()
    # channelOut.packetSeq          = simmods.errorConceal(channelOut.packetSeq, loss, ['interpolation', 'linear'])
    # total_time = time.time() - start_time
    # print(f"Error concealment completed in {total_time}")

    # start_time = time.time()
    # qnt = QL(8)
    # qnt.bitQuantizer(channelOut.packetSeq)
    # channelOut.packetSeq = qnt.inverseQuantizer()
    # total_time = time.time() - start_time
    # print(f"Quantization complete in {total_time}!!")
    #
    # start_time = time.time()
    # remoteOut            = simmods.remoteSim(testModel.remoteModel, channelOut)
    # total_time = time.time() - start_time
    # print(f"Remote Simulation complete in {total_time}!!")
    # start_time = time.time()
    # classValues = np.full(len(filenames), 235)  #push this to utils
    # predictions = np.argmax(remoteOut, axis=1)
    # loss = errorCalc(predictions, classValues)
    # print(loss)
    # t = time.time() - a
    # print(f"Time for one simulation {t}")
    # print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    runSimulation()
