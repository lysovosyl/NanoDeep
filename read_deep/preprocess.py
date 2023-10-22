import numpy as np
import os


def signal_preprocess(signal):
    signal = np.array(signal)
    signal = (signal - np.average(signal)) / np.var(signal)
    return signal