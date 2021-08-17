from copy import deepcopy

import numpy as np


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
    def getActionValues(self, s):
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)

        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        q_vals = np.dot(x, W1) + b1

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
