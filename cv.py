import numpy as np
import cv2 as cv
import mss
import mss.tools
from PIL import Image

# === environment constants ===
MON_NUM = 2

CAPTURE_WIDTH = 440
CAPTURE_HEIGHT = 480

CAPTURE_GRID_STEP = 40

CAPTURE_OFFSET = {
    "left": 20,
    "top": 60,
}

COLOR_MAPPING = {
    48: 0,  # base
    170: 1,  # body
    118: 2,  # head
    112: 3,  # orb
}

# === variables ===
reading = np.zeros((11, 11))

# capture screen, update reading
with mss.mss() as sct:
    # get monitor info, configure capture
    mon = sct.monitors[MON_NUM]
    monitor = {
        "height": CAPTURE_HEIGHT,
        "left": int(mon["left"] + mon["width"] / 2 - CAPTURE_WIDTH / 2),
        "mon": MON_NUM,
        "top": int(mon["height"] / 2 - CAPTURE_HEIGHT / 2),
        "width": CAPTURE_WIDTH,
    }

    while True:
        # take screenshot
        screenshot = sct.grab(monitor)
        img = cv.cvtColor(
            np.array(
                Image.frombytes(
                    "RGB", (screenshot.width, screenshot.height), screenshot.rgb
                )
            ),
            cv.COLOR_RGB2GRAY,
        )

        # read screenshot pixels, update reading
        for row in range(11):
            for col in range(11):

                reading[col][row] = COLOR_MAPPING[
                    img
                    [CAPTURE_OFFSET["top"] + col * CAPTURE_GRID_STEP]
                    [CAPTURE_OFFSET["left"] + row * CAPTURE_GRID_STEP]
                ]

        # debug viewing
        print(reading)
        cv.imshow("", img)

        ## break
        if cv.waitKey(33) & 0xFF in (ord("q"), 27):
            break
