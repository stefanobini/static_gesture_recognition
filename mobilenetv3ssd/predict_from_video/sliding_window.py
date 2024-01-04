import numpy as np


class SlidingWindow:
    def __init__(self, window_size:int, element_size:int):
        self.window_size = window_size
        self.window = np.zeros((window_size,element_size))
        self.insert_idx = 0

    def put(self, element):
        self.window[self.insert_idx] = element
        self.insert_idx += 1
        self.insert_idx = 0 if self.insert_idx == self.window_size else self.insert_idx

    def get_max(self):
        avg_window = np.mean(self.window, axis=0)
        return np.argmax(avg_window), np.amax(avg_window)

    def get_window(self):
        return self.window