import argparse
import re
import yaml
import sys
import os
from talloc import taskAllocater
from spimulation.testConfig import runSimulation
from download.utils import downloadModel

class ParserError(Exception):
    """Used to throw exceptions when multiple options are selected by
       user.
    """
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

def isURL(s):
    """Checks if the given string is a valid http/https url.

    # Arguments
        s: Input string, can be a url

    # Returns
        bool value stating whether the given string is a url
    """
    url = re.compile("""http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|
                        (?:%[0-9a-fA-F][0-9a-fA-F]))+""")
    return bool(url.match(s))

def selectParamConfig(p, paramDict):
    """Throws everything except for the user selected parameter.

    # Argument
        p: parameter for which selection is being made
        paramDict: dictionary containing user selected parameters

    # Returns
        Selected parameter and the corresponding values

    # Raises
        ParserError: if more than one option is selected for a parameter
    """
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
    """Refines the parameter dictionary to only include user selected parameters.

    # Arguments
        config: dictionary read from the YAML file

    # Returns
        Dictionary containing only the options selected by the user
    """
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
    """Called by main function, is responsible for reading in the YAML config file
       and calling the corresponding functions.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--params", help="path to the config file containing parameters", required=True)

    args = vars(ap.parse_args())

    fileName = args['params']

    with open(fileName) as c:
        config = yaml.load(c)
    paramsDict = configSettings(config)

    model = paramsDict['Model']['kerasmodel']
    modelDict = {'xception':'Xception', 'vgg16':'VGG16', 'VGG19':'VGG19', 'resnet50':'ResNet50',
                 'inceptionv3':'InceptionV3', 'inceptionresnetv2':'InceptionResnetV2',
                 'mobilenet':'MobileNet', 'densenet':'DenseNet','nasnet':'NASNet'}

    if isURL(model):
        model = downloadModel(model)
    elif model.lower() in modelDict:
        model = modelDict[model.lower()]
    else:
        print('Unable to load the given model!')
        sys.exit(0)

    task = paramsDict['Task']['value']

    task = taskAllocater(task, paramsDict['TestInput']['testdir'],
                        paramsDict['PreProcess']['reshapeDims'],
                        paramsDict['PreProcess']['batch_size'],
                        paramsDict['PreProcess']['normalize'])

    epoch = paramsDict['Task']['epochs']
    splitLayer = paramsDict['SplitLayer']['split']
    transDict  = paramsDict['Transmission']

    simDir = paramsDict['OutputDir']['simDataDir']

    if not os.path.exists(simDir):
        try:
            os.makedirs(simDir)
        except OSError as exc:
            if exc.errno != errno.EXIST:
                raise

    runSimulation(model, epoch, splitLayer, task, modelDict, transDict, simDir)


if __name__ == "__main__":
    userInterface()
