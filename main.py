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
        'stateCount': 121,
        'hiddenUnitCount': [256, 128, 64, 16],
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

    # start screen capture
    readingThread.start()

    # print ready up prompt
    print('move your cursor to window')
    time.sleep(1)
    print('training starts in 3')
    time.sleep(1)
    print('2')
    time.sleep(1)
    print('1')

    # start training
    reading, orbPos, headPos, gameover = reader.getState()
    agent.agent_start(np.array(reading).flatten())
    while True:
        tempReading, tempOrbPos, headPos, gameover = reader.getState()
        # reset if gameover
        if gameover:
            agent.agent_end(-10000)
            controller.apply(4)
            reading, orbPos, headPos, gameover = reader.getState()

            agent.agent_start(np.array(reading).flatten())

        # step controller
        elif not np.array_equal(reading, tempReading):
            reading = tempReading
            reward = -1

            if orbPos != tempOrbPos:
                orbPos = tempOrbPos
                reward = 20

            action = agent.agent_step(reward, np.array(reading).flatten())
            controller.apply(action)

            print(reading)


if __name__ == "__main__":
    main()
