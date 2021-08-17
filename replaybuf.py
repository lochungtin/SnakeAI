import numpy as np

class ReplayBuffer:
    def __init__(self, size, minibatch_size):
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState()
        self.max_size = size

    # append new state, remove oldest if full
    def append(self, state, action, reward, terminal, next_state):
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    # get random sample
    def getSample(self):
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    # get buffer size
    def getSize(self):
        return len(self.buffer)
