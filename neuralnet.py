from copy import deepcopy
import json
import numpy as np
import os


class NeuralNetwork:
    def __init__(self, stateCount, hiddenUnitCount, actionCount):
        self.stateCount = stateCount
        self.hiddenUnitCount = hiddenUnitCount
        self.actionCount = actionCount

        # initialize layers
        self.layerSize = [stateCount]
        for count in hiddenUnitCount:
            self.layerSize.append(count)

        self.layerSize.append(actionCount)

        # initialize weights
        self.weights = [dict() for i in range(0, len(self.layerSize) - 1)]
        for i in range(0, len(self.layerSize) - 1):
            self.weights[i]['W'] = self.genWeights(
                self.layerSize[i], self.layerSize[i + 1]
            )
            self.weights[i]['b'] = np.zeros((1, self.layerSize[i + 1]))

    # initialize random weight for __init__ with normal dist
    def genWeights(self, rows, cols):
        tensor = np.random.RandomState().normal(0, 1, (rows, cols))

        if rows < cols:
            tensor = tensor.T

        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T

        return tensor

    # get Q value
    def getActionValues(self, state):
        layers = len(self.layerSize) - 2
        x = state
        for i in range(layers):
            w, b = self.weights[i]['W'], self.weights[i]['b']
            psi = np.dot(x, w) + b

            # relu
            x = np.maximum(psi, 0)

        w, b = self.weights[layers]['W'], self.weights[layers]['b']
        q_vals = np.dot(x, w) + b

        return q_vals

    # get temporal difference update
    def getTDUpdate(self, s, delta_mat):
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        W1 = self.weights[1]['W']

        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)
        dx = (psi > 0).astype(float)

        td_update = [dict() for i in range(len(self.weights))]

        v = delta_mat
        td_update[1]['W'] = np.dot(x.T, v) * 1. / s.shape[0]
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]

        v = np.dot(v, W1.T) * dx
        td_update[0]['W'] = np.dot(s.T, v) * 1. / s.shape[0]
        td_update[0]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]

        return td_update

    # get and set weights for updating
    def getWeights(self):
        return deepcopy(self.weights)

    def setWeight(self, weights):
        self.weights = deepcopy(weights)

    # save and load NN config
    def save(self):
        weights = deepcopy(self.weights)

        for i in weights:
            i['W'] = i['W'].tolist()
            i['b'] = i['b'].tolist()

        dictionary = {
            'actionCount': self.actionCount,
            'hiddenUnitCount': self.hiddenUnitCount,
            'layerSize': self.layerSize,
            'stateCount': self.stateCount,
            'weights': weights
        }

        file = open('nnconfig.json', 'w')
        json.dump(dictionary, file)

        file.close()

    def load(self):
        file = open('nnconfig.json', 'r')
        dictionary = file.read()

        self.actionCount = dictionary['actionCount']
        self.hiddenUnitCount = dictionary['hiddenUnitCount']
        self.stateCount = dictionary['stateCount']
        self.layerSize = dictionary['layerSize']
        
        for i in dictionary['weights']:
            i['W'] = np.array(i['W'])
            i['b'] = np.array(i['b'])

        self.weights = dictionary['weights']

        file.close()
