import numpy as np


class Adam:
    def __init__(self, layerSizes, stepSize, bM, bV, epsilon):
        # initialize hyperparams
        self.layerSizes = layerSizes
        self.stepSize = stepSize
        self.bM = bM
        self.bV = bV
        self.epsilon = epsilon

        self.m = [dict() for i in range(1, len(self.layerSizes))]
        self.v = [dict() for i in range(1, len(self.layerSizes))]

        for i in range(0, len(self.layerSizes) - 1):
            self.m[i]["W"] = np.zeros(
                (self.layerSizes[i], self.layerSizes[i + 1])
            )
            self.m[i]["b"] = np.zeros((1, self.layerSizes[i + 1]))

            self.v[i]["W"] = np.zeros(
                (self.layerSizes[i], self.layerSizes[i + 1])
            )
            self.v[i]["b"] = np.zeros((1, self.layerSizes[i + 1]))

        self.betaMProd = self.bM
        self.betaVProd = self.bV

    # update weights with the adam optimizer
    def updateWeights(self, weights, tdErrXGrad):
        for i in range(len(weights)):
            for param in weights[i].keys():
                self.m[i][param] = self.bM * self.m[i][param] + (
                    1 - self.bM) * tdErrXGrad[i][param]

                self.v[i][param] = self.bV * self.v[i][param] + (
                    1 - self.bV) * tdErrXGrad[i][param] ** 2

                mHat = self.m[i][param] / (1 - self.betaMProd)
                vHat = self.v[i][param] / (1 - self.betaVProd)

                weights[i][param] += self.stepSize * \
                    mHat / (np.sqrt(vHat) + self.epsilon)

        self.betaMProd *= self.bM
        self.betaVProd *= self.bV

        return weights
