import sys
sys.path.append('..')

from channel.gbChannel import GBC
from channel.trivialChannel import RLC
from models.quantizer import QLayer as QL
from plc import linearInterp, nearestNeighbours

def quantInit(quantization):
    if quantization['include']:
        return QL(quantization['numberOfBits'])
    else:
        return 'noQuant'

def loadChannel(channel):
    chtype = list(channel.keys())[0]

    if chtype==0:
        return 'noChannel'
    elif chtype=='GilbertChannel':
        lp = channel[chtype]['lossProbability']
        bl = channel[chtype]['burstLength']
        return GBC(lp, bl)
    elif chtype=='RandomLossChannel':
        lp = channel[chtype]['lossProbability']
        return RLC(lp)

def plcLoader(lossConceal):
    chtype = list(lossConceal.keys())[0]

    if chtype == 0:
        return 'noConceal'
    elif chtype== 'Linear':
        return linearInterp.interpPackets
    elif chtype== 'nearestNeighbours':
        return nearestNeighbours.interpPackets
