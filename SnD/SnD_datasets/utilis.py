

import torch
import csv
import os
import re
import json
import numpy as np
from settings import parse_args

from info import TASK2INFO


def rolling_window(a, window): 
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    c = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return c

def vview(a): 
    return np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))


def sublist_start_index(a, b):
    n = min(len(b), len(a))
    target_lists = [rolling_window(np.array(a), i) for i in range(1, n + 1)]
    res = [np.flatnonzero(vview(target_lists[-1]) == s)
           for s in vview(np.array([b]))][0]
    if len(res) == 0:
        k = 3
        for i in range(1,k):
            return sublist_start_index(a, b[:-i])
    else:
        return res[0]



def pad_seq(seq, pad, max_len, pad_left=False):
    if pad_left:
        return [pad] * (max_len - len(seq)) + seq
    else:
        return seq + [pad] * (max_len - len(seq))