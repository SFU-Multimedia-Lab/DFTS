import argparse
import re
import yaml
import sys
from talloc import taskAllocater
from spimulation.testConfig import runSimulation
from download.utils import downloadModel

class ParserError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

def isURL(s):
    #check if given string is url or not
    url = re.compile("""http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|
                        (?:%[0-9a-fA-F][0-9a-fA-F]))+""")
    return bool(url.match(s))

def selectParamConfig(p, paramDict):
    sum = 0
    index = 0

    for i in paramDict:
        sum += paramDict[i]['include']
        if paramDict[i]['include']:
            index = i
    try:
        if sum>1:
            raise ParserError(f"Multiple configurations selected for {p}", sum)
    except Exception as e:
        raise
    else:
        if sum==0:
            return (index, False)
        else:
            return (index, paramDict[index])

def configSettings(config):
    for i in config:
        if i=='Transmission':
            for j in config[i]:
                transDict = {}
                if j=='channel' or j=='concealment':
                    index, temp = selectParamConfig(j, config[i][j])
                    transDict[index] = temp
                    config[i][j] = transDict
    return config


def userInterface():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--params", help="path to the config file containing parameters", required=True)

    args = vars(ap.parse_args())

    fileName = args['params']

    with open(fileName) as c:
        config = yaml.load(c)
    paramsDict = configSettings(config)

    model = paramsDict['Model']['kerasmodel']

    if isURL(model):
        model = downloadModel(model)
    else:
        modelDict = {'xception':'Xception', 'vgg16':'VGG16', 'VGG19':'VGG19', 'resnet50':'ResNet50',
                     'inceptionv3':'InceptionV3', 'inceptionresnetv2':'InceptionResnetV2',
                     'mobilenet':'MobileNet', 'densenet':'DenseNet','nasnet':'NASNet'}
        model = modelDict[model.lower()]
    else:
        print('Unable to load the given model!')
        sys.exit(0)

    task = paramDict['Task']['value']

    task = taskAllocater(task, paramsDict['TestInput']['testdir'],
                        paramsDict['PreProcess']['reshapeDims'],
                        paramsDict['PreProcess']['batch_size'],
                        paramsDict['PreProcess']['normalize'])

    splitLayer = paramsDict['SplitLayer']['split']
    transDict  = paramsDict['Transmission']


if __name__ == "__main__":
    userInterface()
