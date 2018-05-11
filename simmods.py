from utils import preprocess
import sys

modelDict = {'xception':'Xception', 'vgg16':'vgg16', 'VGG19':'vgg19', 'resnet50':'ResNet50',
             'inceptionv3':'InceptionV3', 'inceptionresnetv2':'InceptionResnetV2',
             'mobilenet':'MobileNet', 'densenet':'DenseNet','nasnet':'NASNet'}

def deviceSim(model, layer, testImage):
    '''
        * simulate a user device
        * runs the image through the pretrained model up until the specified layer number
    '''
    # print(modelDict)
    # return (2,3)
    try:
        model = modelDict[model.lower()]
    except KeyError:
        print("We currently do not support that model. Please try a valid one.")
        sys.exit(1)
    preprocessedImage = preprocess(testImage, model)
    print(preprocessedImage)
    return (2,3)

def compress(deviceOut):
    pass

def transmit():
    pass

def remoteSim(channelOut):
    pass
