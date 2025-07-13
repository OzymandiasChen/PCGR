
import torch

from collections import OrderedDict
from info import IDTOTASK



class Memory(object):
    def __init__(self, buffer=None):
        if buffer is None:
            self.memory = {}
            print('memory model: current memory has saved %d tasks' %
                  len(self.memory.keys()), flush=True)
            total_keys = len(self.memory.keys())
        else:
            self.memory = buffer.memory
            total_keys = len(self.memory.keys())

    def push(self, task_name, value):
        self.memory[task_name] = value
    
    def memory_update(self, task_name, value):
        return 


class Memory_std(object):
    def __init__(self, buffer=None, path=None):
        if buffer is None:
            self.memory = OrderedDict()
        else:
            self.memory = buffer.memory
        
        if path is not None:
            self.memory = torch.load(path)

    def print(self, logger):
        return

    def save(self, path):
        torch.save(self.memory, path)
    
    def load(self, path):
        self.memory = torch.load(path)
    
