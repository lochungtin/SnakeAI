from copy import deepcopy
from datetime import datetime
import json
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

    # get TD error * gradient
    def getTDUpdate(self, states , deltaMatrix):
        layers = len(self.weights)
        bS = states.shape[0]
        tdUpdate = [dict() for i in range(layers)]

        inputs = [states]
        dxS = []

        for i in range(layers):
            w, b = self.weights[i]['W'], self.weights[i]['b']
            psi = np.dot(inputs[i], w) + b

            # x
            inputs.append(np.maximum(psi, 0))
            # dx
            dxS.append((np.dot(inputs[i], w) + b > 0).astype(float))       

        vS = [None for i in range(layers)]
        vS[layers - 1] = deltaMatrix
        for i in range(layers - 2, -1, -1):
            # v
            vS[i] = np.dot(vS[i + 1], self.weights[i + 1]['W'].T) * dxS[i]

        for i in range(layers):
            tdUpdate[i]['W'] = np.dot(inputs[i].T, vS[i]) * bS
            tdUpdate[i]['b'] = np.sum(vS[i], axis=0, keepdims=True) / bS
            
        return tdUpdate

    # get and set weights for updating
    def getWeights(self):
        return deepcopy(self.weights)

    def setWeights(self, weights):
        self.weights = deepcopy(weights)

    # save and load NN config
    def save(self, epCount):
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

        fileName = './out/' + datetime.now().strftime('%d:%m:%Y:%H:%M:%S_') + 'ep' + str(epCount) + '.json';

        file = open(fileName, 'w')
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
