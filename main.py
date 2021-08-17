import numpy as np
from reader import Reader
import threading

def main():
    # start screen capture
    reader = Reader()

    readingThread = threading.Thread(target=reader.start)
    readingThread.start()

    while True:
        print(reader.get_readings(), reader.get_gameover())

if __name__ == "__main__":
    main()