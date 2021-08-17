from copy import deepcopy
import numpy as np

from adam import Adam
from neuralnet import NeuralNetwork
from replaybuf import ReplayBuffer
from utils import Utils


class Agent:
    def __init__(self, config):
        # init agent params
        self.replay_buffer = ReplayBuffer(
            config['rbConfig']['rbSize'],
            config['rbConfig']['batchSize'],
        )
        self.num_replay = config['rbConfig']['replayUpdatePerStep']

        self.network = NeuralNetwork(
            config['nnConfig']['stateCount'],
            config['nnConfig']['hiddenUnitCount'],
            config['nnConfig']['actionCount'],
        )
        self.optimizer = Adam(
            self.network.layerSize,
            config['adamConfig']['stepSize'],
            config['adamConfig']['betaM'],
            config['adamConfig']['betaV'],
            config['adamConfig']['epsilon'],
        )

        self.num_actions = config['nnConfig']['actionCount']
        self.discount = config['gamma']
        self.tau = config['tau']

        self.rand_generator = np.random.RandomState()

        self.util = Utils()

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0

    # choose action according to policy and softmax
    def policy(self, state):
        return self.rand_generator.choice(
            a=self.num_actions,
            p=self.util.softmax(
                self.network.getActionValues(
                    state
                ), self.tau
            ).squeeze(),
        )

    # start episode
    def agent_start(self, state):
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)

        return self.last_action

    # time step increment 
    def agent_step(self, reward, state):
        self.sum_rewards += reward
        self.episode_steps += 1

        state = np.array([state])

        action = self.policy(state)

        self.replay_buffer.append(
            self.last_state,
            self.last_action,
            reward,
            False,
            state
        )

        if self.replay_buffer.getSize() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for i in range(self.num_replay):
                self.util.optimize_network(
                    self.replay_buffer.getSample(),
                    self.discount,
                    self.optimizer,
                    self.network,
                    current_q,
                    self.tau
                )

        self.last_state = state
        self.last_action = action

        return action

    # end episode
    def agent_end(self, reward):
        self.sum_rewards += reward
        self.episode_steps += 1

        state = np.zeros_like(self.last_state)

        self.replay_buffer.append(
            self.last_state,
            self.last_action,
            reward,
            True,
            state
        )

        if self.replay_buffer.getSize() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for i in range(self.num_replay):
                self.util.optimize_network(
                    self.replay_buffer.getSample(),
                    self.discount,
                    self.optimizer,
                    self.network,
                    current_q,
                    self.tau
                )

    def saveNN(self):
        self.network.save()

    def loadNN(self):
        self.network.load()
