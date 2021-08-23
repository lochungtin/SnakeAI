import numpy as np
import cv2 as cv
import mss
import mss.tools
from PIL import Image
from numpy.core.fromnumeric import shape


# === environment constants ===
CAPTURE_WIDTH = 440
CAPTURE_HEIGHT = 470

CAPTURE_GRID_STEP = 40

CAPTURE_OFFSET = {
    'left': 20,
    'top': 42,
}

CAPTURE_DIRECTION_PIXELS = [(5, 190), (5, 210), (5, 230), (5, 250)]
CAPTURE_DIRECTION_LVALUE = 170

CAPTURE_GAMEOVER_PIXEL = (5, 410)
CAPTURE_GAMEOVER_LVALUE = 170

CAPTURE_NEW_ORB_PIXEL = (5, 430)
CAPTURE_NEW_ORB_LVALUE = 112

COLOR_MAPPING = {
    48: (BASE_INDX := 0),   # base
    170: (BODY_INDX := 1),  # body
    118: (HEAD_INDX := 2),  # head
    112: (ORB_INDX := 3),   # orb
}
MAX_GRID_DIM = 11


class Reader:
    def __init__(self):
        self.monNum = 0

        self.captureConfig = None
        self.gridDim = 11

        self.reading = None
        self.direction = 0
        self.orbPos = None
        self.headPos = None

        self.gameover = False
        self.newOrb = False

        self.printDebug = False
        self.showWindow = True

    # get state
    def getState(self):
        orbX, orbY = self.orbPos
        snakeX, snakeY = self.headPos

        state = [
            int(orbY < snakeY),  # is orb up
            int(orbY > snakeY),  # is orb down
            int(orbX < snakeX),  # is orb left
            int(orbX > snakeY),  # is orb right
            # has obstacle up
            int(snakeY == 0 or self.reading[snakeX][snakeY - 1] == BODY_INDX),
            # has obstacle down
            int(snakeY == (self.gridDim - 1)
                or self.reading[snakeX][snakeY + 1] == BODY_INDX),
            # has obstacle left
            int(snakeX == 0 or self.reading[snakeX - 1]
                [snakeY] == BODY_INDX),
            # has obstacle right
            int(snakeX == (self.gridDim - 1)
                or self.reading[snakeX + 1][snakeY] == BODY_INDX),
            # snake direction
            int(self.direction == 0),
            int(self.direction == 1),
            int(self.direction == 2),
            int(self.direction == 3),
        ]

        return state, abs(orbX - snakeX) + abs(orbY - snakeY), self.newOrb, self.gameover

    # select monitor
    # must call when using dual monitors
    def selectMonitor(self, monNum):
        self.monNum = monNum

    # find grid position and dim
    def calibrate(self):
        with mss.mss() as sct:
            mon = sct.monitors[self.monNum]

            fullCapDim = {
                'height': mon['height'],
                'left': mon['left'],
                'mon': self.monNum,
                'top': mon['top'],
                'width': mon['width'],
            }

            # find position circle
            screenshot = sct.grab(fullCapDim)
            img = np.array(
                Image.frombytes(
                    'RGB', (screenshot.width,
                            screenshot.height), screenshot.rgb
                )
            )

            found = False
            x, y = 0, 0
            for i in range(mon['height']):
                for j in range(mon['width']):
                    if (img[i][j] == (162, 252, 225)).all():
                        x, y = j, i + 5
                        found = True
                        break
                if found:
                    break

            self.captureConfig = {
                'height': CAPTURE_HEIGHT,
                'left': mon['left'] + x - 9,
                'mon': self.monNum,
                'top': mon['top'] + y - 5,
                'width': CAPTURE_WIDTH,
            }

            # read grid dimension
            screenshot = sct.grab(self.captureConfig)
            img = cv.cvtColor(
                np.array(
                    Image.frombytes(
                        'RGB', (screenshot.width,
                                screenshot.height), screenshot.rgb
                    )
                ),
                cv.COLOR_RGB2GRAY,
            )

            count = 0
            for col in range(11):
                for row in range(11):
                    val = img[CAPTURE_OFFSET['top'] + row *
                              CAPTURE_GRID_STEP][CAPTURE_OFFSET['left'] + col * CAPTURE_GRID_STEP]

                    if np.average(val) != 30:
                        count += 1

            self.gridDim = int(np.sqrt(count))
            self.reading = np.zeros((self.gridDim, self.gridDim))

    # main
    def start(self):
        # capture screen, update reading
        with mss.mss() as sct:
            while True:
                # take screenshot
                screenshot = sct.grab(self.captureConfig)
                img = cv.cvtColor(
                    np.array(
                        Image.frombytes(
                            'RGB', (screenshot.width,
                                    screenshot.height), screenshot.rgb
                        )
                    ),
                    cv.COLOR_RGB2GRAY,
                )

                # read screenshot pixels, update reading
                pxlInc = int(2 - (self.gridDim - 7) / 2)
                for col in range(self.gridDim):
                    for row in range(self.gridDim):
                        encoding = COLOR_MAPPING[
                            img
                            [CAPTURE_OFFSET['top'] +
                                (row + pxlInc) * CAPTURE_GRID_STEP]
                            [CAPTURE_OFFSET['left'] +
                                (col + pxlInc) * CAPTURE_GRID_STEP]
                        ]
                        self.reading[row][col] = encoding

                        if encoding == 3:
                            self.orbPos = (row, col)
                        elif encoding == 2:
                            self.headPos = (row, col)

                # update gameover boolean
                self.gameover = img[CAPTURE_GAMEOVER_PIXEL] == CAPTURE_GAMEOVER_LVALUE

                # update new orb boolean
                self.newOrb = img[CAPTURE_NEW_ORB_PIXEL] == CAPTURE_NEW_ORB_LVALUE

                # update direction value
                for index, pos in enumerate(CAPTURE_DIRECTION_PIXELS):
                    if img[pos] == CAPTURE_DIRECTION_LVALUE:
                        self.direction = index

                # debug viewing
                self.printDebug and print(self.reading)
                self.showWindow and cv.imshow('', img)

                # break
                if cv.waitKey(33) & 0xFF in (ord('q'), 27):
                    break
