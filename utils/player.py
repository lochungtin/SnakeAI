import numpy as np

from utils.neuralnet import NeuralNetwork


class Player:
    def __init__(self, confFileName):
        self.network = NeuralNetwork(0, [0, 0, 0], 0)
        self.network.load(confFileName)

    def getAction(self, state):
        return np.argmax(self.network.getActionValues(state))