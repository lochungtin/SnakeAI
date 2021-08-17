import numpy as np


class Utils:
    # softmax function
    def softmax(self, action_values, tau=1.0):
        preferences = action_values / tau
        max_preference = np.max(preferences, axis=1).reshape((-1, 1))

        exp_perferences = np.exp(preferences - max_preference)
        sum_of_exp_preferences = np.sum(exp_perferences, axis=1)

        return (exp_perferences / sum_of_exp_preferences.reshape((-1, 1))).squeeze()

    # get temporal difference error
    def getTDError(self, states, nState, actions, rewards, discount, terminals, network, curQ, tau):
        q_next_mat = curQ.getActionValues(nState)
        probs_mat = self.softmax(q_next_mat, tau)
        v_next_vec = np.sum(q_next_mat * probs_mat, axis=1) * (1 - terminals)
        target_vec = rewards + discount * v_next_vec

        q_mat = network.getActionValues(states)
        batch_indices = np.arange(q_mat.shape[0])
        q_vec = q_mat[batch_indices, actions]

        return target_vec - q_vec

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

        delta_vec = self.getTDError(
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

        batch_indices = np.arange(batchSize)

        delta_mat = np.zeros((batchSize, network.actionCount))
        delta_mat[batch_indices, actions] = delta_vec

        td_update = network.getTDUpdate(states, delta_mat)

        weights = optimizer.update_weights(network.get_weights(), td_update)

        network.set_weights(weights)
