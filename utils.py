import numpy as np


class Utils:
    # softmax function
    def softmax(self, action_values, tau=1.0):
        pref = action_values / tau
        maxPref = np.max(pref, axis=1).reshape((-1, 1))

        ePref = np.exp(pref - maxPref)
        sumEPref = np.sum(ePref, axis=1)

        return (ePref / sumEPref.reshape((-1, 1))).squeeze()

    # get temporal difference error
    def getTDError(self, states, nState, actions, rewards, discount, terminals, network, curQ, tau):
        qNMatrix = curQ.getActionValues(nState)
        probsMatrix = self.softmax(qNMatrix, tau)
        vNVector = np.sum(qNMatrix * probsMatrix, axis=1) * (1 - terminals)
        targetVector = rewards + discount * vNVector

        qMatrix = network.getActionValues(states)
        batchIndxs = np.arange(qMatrix.shape[0])
        qVector = qMatrix[batchIndxs, actions]

        return targetVector - qVector

    # optimize the neural network
    def optimizeNN(self, experiences, discount, optimizer, network, curQ, tau):
        states, actions, rewards, terminals, nState = map(
            list,
            zip(*experiences),
        )
        states = np.concatenate(states)
        nState = np.concatenate(nState)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        batchSize = states.shape[0]

        deltaVector = self.getTDError(
            states,
            nState,
            actions,
            rewards,
            discount,
            terminals,
            network,
            curQ,
            tau,
        )

        batchIndxs = np.arange(batchSize)

        deltaMatrix = np.zeros((batchSize, network.actionCount))
        deltaMatrix[batchIndxs][actions] = deltaVector

        tdUpdate = network.getTDUpdate(states, deltaMatrix)

        weights = optimizer.update_weights(network.get_weights(), tdUpdate)

        network.set_weights(weights)
