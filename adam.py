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

        self.beta_m_product = self.bM
        self.beta_v_product = self.bV

    # update weights with the adam optimizer
    def update_weights(self, weights, td_errors_times_gradients):
        for i in range(len(weights)):
            for param in weights[i].keys():
                self.m[i][param] = self.bM * self.m[i][param] + (
                    1 - self.bM) * td_errors_times_gradients[i][param]

                self.v[i][param] = self.bV * self.v[i][param] + (
                    1 - self.bV) * td_errors_times_gradients[i][param] ** 2

                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                weights[i][param] += self.stepSize * \
                    m_hat / (np.sqrt(v_hat) + self.epsilon)

        self.beta_m_product *= self.bM
        self.beta_v_product *= self.bV

        return weights
