import numpy as np
import reader
import threading

def main():
    # start screen capture
    readingThread = threading.Thread(target=reader.start)
    readingThread.start()

    while True:
        print(reader.get_readings(), reader.get_gameover())

if __name__ == "__main__":
    main()