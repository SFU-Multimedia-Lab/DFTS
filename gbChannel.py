import random

class GBC(object):
    """docstring for GBC."""
    def __init__(self, lossProb, burstLength):
        super(GBC, self).__init__()
        self.lp = lossProb #fixed initially
        self.bl = burstLength
        self.state = 0 #initially bad channel state
        self.calcChannelProb()

    def calcChannelProb(self):
        self.pbg = 1.0/self.bl
        self.pgb = self.pbg/((1.0/self.lp)-1)

    def simulate(self, nofSims):
        for i in range(nofSims):
            self.flip(self.state)

    def flip(self, state):
        print(f"{state}", end=" ")
        if state==1:
            p = random.gauss(0, 1)
            if p>self.pgb:
                self.state = 0
                return
            return
        else:
            p = random.random()
            if p>self.pbg:
                self.state = 1
                return
            return

def runChannelSim():
    lossProb = 0.05
    burstLength = 3
    p = GBC(lossProb, burstLength)
    p.simulate(100)

if __name__ == '__main__':
    runChannelSim()
