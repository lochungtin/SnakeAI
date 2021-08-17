from copy import deepcopy
import numpy as np

from adam import Adam
from neuralnet import NeuralNetwork
from replaybuf import ReplayBuffer
from utils import Utils


class Agent:
    def __init__(self, config):
        # init agent params
        self.replayBuffer = ReplayBuffer(
            config['rbConfig']['rbSize'],
            config['rbConfig']['batchSize'],
        )
        self.replayCount = config['rbConfig']['replayUpdatePerStep']

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

        self.actionCount = config['nnConfig']['actionCount']
        self.discount = config['gamma']
        self.tau = config['tau']

        self.rand_generator = np.random.RandomState()

        self.util = Utils()

        self.pState = None
        self.pAction = None

        self.rSum = 0
        self.epSteps = 0

    # choose action according to policy and softmax
    def policy(self, state):
        return self.rand_generator.choice(
            a=self.actionCount,
            p=self.util.softmax(
                self.network.getActionValues(
                    state
                ), self.tau
            ).squeeze(),
        )

    # start episode
    def start(self, state):
        self.rSum = 0
        self.epSteps = 0
        self.pState = np.array([state])
        self.pAction = self.policy(self.pState)

        return self.pAction

    # time step increment 
    def step(self, reward, state):
        self.rSum += reward
        self.epSteps += 1

        state = np.array([state])

        action = self.policy(state)

        self.replayBuffer.append(
            self.pState,
            self.pAction,
            reward,
            False,
            state
        )

        if self.replayBuffer.getSize() > self.replayBuffer.miniBatchSize:
            current_q = deepcopy(self.network)
            for i in range(self.replayCount):
                self.util.optimizeNN(
                    self.replayBuffer.getSample(),
                    self.discount,
                    self.optimizer,
                    self.network,
                    current_q,
                    self.tau
                )

        self.pState = state
        self.pAction = action

        return action

    # end episode
    def end(self, reward):
        self.rSum += reward
        self.epSteps += 1

        state = np.zeros_like(self.pState)

        self.replayBuffer.append(
            self.pState,
            self.pAction,
            reward,
            True,
            state
        )

        if self.replayBuffer.getSize() > self.replayBuffer.miniBatchSize:
            current_q = deepcopy(self.network)
            for i in range(self.replayCount):
                self.util.optimizeNN(
                    self.replayBuffer.getSample(),
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
