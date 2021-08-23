from pynput.keyboard import Key
import numpy as np
import threading
import time

from agent import Agent
from keyboard import Keyboard
from reader import Reader


# env vars
orbPos = None
reading = None

# screen reader
reader = Reader()
reader.showWindow = False
reader.printDebug = False
readingThread = threading.Thread(target=reader.start)

# agent
agent = Agent({
    'adamConfig': {
        'stepSize': 1e-3,
        'betaM': 0.9,
        'betaV': 0.999,
        'epsilon': 1e-8
    },
    'nnConfig': {
        'stateCount': 14,
        'hiddenUnitCount': [128, 64, 16],
        'actionCount': 4,
    },
    'rbConfig': {
        'rbSize': 50000,
        'batchSize': 8,
        'replayUpdatePerStep': 4,
    },
    'gamma': 0.99,
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
    reading, orbPos, headPos, gameover = reader.getState()
    agent.start(np.array(reading).flatten())

    eps = 0

    while True:
        tempReading, tempOrbPos, headPos, gameover = reader.getState()
        # reset if gameover
        if gameover:
            agent.end(-100)
            print('Eps: {} | reward: {}'.format(eps, agent.rSum))
            eps += 1

            time.sleep(0.5)
            controller.apply(4)
            time.sleep(0.1)
            reading, orbPos, headPos, gameover = reader.getState()

            agent.start(np.array(reading).flatten())

        # step controller
        elif not np.array_equal(reading, tempReading):
            reading = tempReading
            reward = 1

            if orbPos != tempOrbPos:
                orbPos = tempOrbPos
                reward = 10

            action = agent.step(reward, np.array(reading).flatten())
            controller.apply(action)


if __name__ == "__main__":
    main()
