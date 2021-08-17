import numpy as np
import threading
import time

from agent import Agent
from neuralnet import NeuralNetwork
from reader import Reader

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


def main():
    # variable binding
    global agent, reading, readingThread

    # start screen capture
    readingThread.start()

    time.sleep(0.5)
    while True:
        temp = reader.get_readings()
        if not np.array_equal(reading, temp):
            reading = temp

            print(reading)


if __name__ == "__main__":
    main()
