from neuralnet import NeuralNetwork
import numpy as np
from reader import Reader
import threading

reading = None


def main():
    # variable binding
    global reading

    # start screen capture
    reader = Reader()

    readingThread = threading.Thread(target=reader.start)
    readingThread.start()

    while True:
        temp = reader.get_readings()
        if not np.array_equal(reading, temp):
            reading = temp

            print(reading)

if __name__ == "__main__":
    main()
