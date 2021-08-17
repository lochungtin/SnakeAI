import numpy as np


class ReplayBuffer:
    def __init__(self, size, miniBatchSize):
        self.buffer = []
        self.miniBatchSize = miniBatchSize
        self.maxSize = size

    # append new state, remove oldest if full
    def append(self, state, action, reward, terminal, nState):
        if len(self.buffer) == self.maxSize:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, nState])

    # get random sample
    def getSample(self):
        idxs = np.random.RandomState().choice(
            np.arange(len(self.buffer)),
            size=self.miniBatchSize,
        )
        return [self.buffer[idx] for idx in idxs]

    # get buffer size
    def getSize(self):
        return len(self.buffer)
