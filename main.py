import argparse

def argumentReader():
    '''
        * Read in command line arguments
            ** m: model such as those given at https://keras.io/applications/
            ** l: layer you want to split your model at
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", help="name of the model", required=True)
    ap.add_argument("-l", "--layer", help="layer you want to split at", required=True)
    args = vars(ap.parse_args())
    return args

def runSimulation():
    args = argumentReader()
    print(args['model'], args['layer'])


if __name__ == '__main__':
    runSimulation()
