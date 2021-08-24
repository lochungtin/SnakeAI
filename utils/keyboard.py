from pynput.keyboard import Controller

class Keyboard:
    def __init__(self, actionMapping):
        self.actionMapping = actionMapping
        self.controller = Controller()

    def apply(self, actionIndex):
        self.controller.press(self.actionMapping[actionIndex])
        self.controller.release(self.actionMapping[actionIndex])
