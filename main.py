from pynput.keyboard import Key
import numpy as np
import threading
import time

from agent import Agent
from keyboard import Keyboard
from reader import Reader


# env vars
orbDist = None

# screen reader
reader = Reader()

reader.showWindow = False
reader.printDebug = False

reader.selectMonitor(2)
reader.calibrate()
readingThread = threading.Thread(target=reader.start)

# agent
agent = Agent({
    'adamConfig': {
        'stepSize': 1e-3,
        'betaM': 0.9,
        'betaV': 0.999,
        'epsilon': 0.001
    },
    'nnConfig': {
        'stateCount': 12,
        'hiddenUnitCount': [128, 64, 16],
        'actionCount': 4,
    },
    'rbConfig': {
        'rbSize': 50000,
        'batchSize': 8,
        'replayUpdatePerStep': 4,
    },
    'gamma': 0.95,
    'tau': 0.001
})

# keyboard controller
controller = Keyboard({
    0: 'w',
    1: 's',
    2: 'a',
    3: 'd',
    4: Key.space,
})


def main():
    # variable binding
    global agent, orbPos, reading, readingThread

    readingThread.start()

    # print ready up prompt
    print('move your cursor to window')
    time.sleep(1)
    print('training starts in\n3')
    time.sleep(1)
    print('2')
    time.sleep(1)
    print('1')

    # start training
    controller.apply(4)
    state, orbDist, newOrb, gameover = reader.getState()
    agent.start(state)

    eps = 0

    while True:
        tempState, newOrbDist, newOrb, gameover = reader.getState()
        # reset if gameover
        if gameover:
            agent.end(-100)
            print('Eps: {} | reward: {}'.format(eps, agent.rSum))
            eps += 1

            if eps % 2 == 0:
                agent.saveNN(eps);

            time.sleep(0.5)
            controller.apply(4)
            time.sleep(0.1)
            state, orbDist, newOrb, gameover = reader.getState()

            agent.start(state)

        # step controller
        elif not np.array_equal(state, tempState) or orbDist != newOrbDist:
            state = tempState
            reward = -1

            if newOrb:
                reward = 10
            elif orbDist > newOrbDist:
                reward = 1

            orbDist = newOrbDist

            print(state, reward)

            action = agent.step(reward, state)
            controller.apply(action)


if __name__ == "__main__":
    main()
