import numpy as np


class LinearAnnealSchedule:
    def __init__(self, start_value, end_value, timeframe):
        assert start_value >= end_value
        assert timeframe > 0
        self.start_value = start_value
        self.end_value = end_value
        self.timeframe = timeframe

    def __call__(self, t):
        delta = self.end_value - self.start_value
        x = self.start_value + (t / self.timeframe) * delta
        return np.clip(x, self.end_value, self.start_value)


class ConstantSchedule(LinearAnnealSchedule):
    def __init__(self, value):
        super().__init__(start_value=value, end_value=value, timeframe=1)
