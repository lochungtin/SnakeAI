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
            config['replay_buffer_size'],
            config['minibatch_sz'],
            config.get("seed"),
        )
        self.num_replay = config['num_replay_updates_per_step']

        self.network = NeuralNetwork(config['network_config'])
        self.optimizer = Adam(
            self.network.layerSize,
            config["optimizer_config"],
        )

        self.num_actions = config['network_config']['num_actions']
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
            self.num_actions,
            p=self.util.softmax(
                self.network.get_action_values(
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

        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for i in range(self.num_replay):
                self.util.optimize_network(
                    self.replay_buffer.sample(),
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

        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for i in range(self.num_replay):
                self.util.optimize_network(
                    self.replay_buffer.sample(),
                    self.discount,
                    self.optimizer,
                    self.network,
                    current_q,
                    self.tau
                )
