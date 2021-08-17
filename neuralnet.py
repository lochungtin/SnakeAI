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
        layers = len(self.weights) - 1
        x = state
        for i in range(layers):
            w, b = self.weights[i]['W'], self.weights[i]['b']
            psi = np.dot(x, w) + b

            # relu
            x = np.maximum(psi, 0)

        w, b = self.weights[layers]['W'], self.weights[layers]['b']
        q_vals = np.dot(x, w) + b

        return q_vals

    # get gradients
    def getGradients(self, state):
        layers = len(self.weights)
        grads = [dict() for i in range(layers)]

        x = np.copy(state)
        for i in range(layers - 1):
            nT = self.weights[i + 1]['W'].T

            # update gradients
            grads[i]['W'] = x.T * np.dot(
                nT,
                nT * (x > 0),
            )
            grads[i]['b'] = nT * (x > 0)

            # calculate next input
            x = np.maximum(
                np.matmul(
                    x,
                    self.weights[i]['W'],
                ) + self.weights[i]['b'], 0,
            )

        return grads

    # get TD error * gradient
    def getTDUpdate(self, state, deltaMatrix):
        layers = len(self.weights)
        tdUpdate = [dict() for i in range(layers)]

        grads = self.getGradients(state)

        for i in range(layers):
            tdUpdate[i]['W'] = deltaMatrix * grads[i]['W']
            tdUpdate[i]['b'] = deltaMatrix * grads[i]['b']

        return tdUpdate

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
