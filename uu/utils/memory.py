import torch
import sys
import psutil
import os
import subprocess


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)




class MemSize:
    def __init__(self, v):
        self.v = v

    def __add__(self, other):
        return self.__class__(self.v + other.v)
    
    def __sub__(self, other):
        return self.__class__(self.v - other.v)
    
    @classmethod
    def fromStr(cls, str):
        suffixes = {'k': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
        if str[-1] in suffixes:
            val = int(float(str[:-1]) * suffixes[str[-1]])
        else:
            val = int(str)
        return MemSize(val)

    def __str__(self):
        return sizeof_fmt(self.v)

    def __format__(self, fmt_spec):
        return sizeof_fmt(self.v).__format__(fmt_spec)
    
    def __repr__(self):
        return str(self.v)

    def __int__(self):
        return self.v

class MeasureMemory:
    def __init__(self, device):
        self.device = device
        self.cuda = self.device.type == 'cuda'
        if not self.cuda:
            self.process = psutil.Process(os.getpid())
            self.max_memory = 0
        self.last_memory = self.currentValue()
        self.start_memory = self.last_memory

    def currentValue(self):
        if self.cuda:
            result = torch.cuda.memory_allocated(self.device)
        else: 
            result = int(self.process.memory_info().rss)
            self.max_memory = max(self.max_memory, result)
        return result
        
    def maximumValue(self):
        if self.cuda:
            return torch.cuda.max_memory_allocated(self.device)
        else:
            return self.max_memory

    def availableValue(self, index=None):
        assert self.cuda
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"])
        l = [int(x) for x in result.strip().split(b"\n")]
        if index is None:
            index = self.device.index
        if index is None: index = torch.cuda.current_device()
        return l[index]*1024*1024 + torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        
    ## Requires Pytorch >= 1.1.0
    def resetMax(self):
        if self.cuda:
            torch.cuda.reset_max_memory_allocated(self.device)
        else:
            self.max_memory = 0
            self.max_memory = self.currentValue()
        

    def current(self):
        return MemSize(self.currentValue())

    def available(self):
        return MemSize(self.availableValue())

    def maxx(self):
        return MemSize(self.maximumValue())
        
    def diffFromLast(self):
        current = self.currentValue()
        result = current - self.last_memory
        self.last_memory = current
        return MemSize(result)

    def diffFromStart(self):
        current = self.currentValue()
        return MemSize(current - self.start_memory)

    def currentCached(self):
        if not self.cuda: 
            return 0
        else: 
            return MemSize(torch.cuda.memory_reserved(self.device))

    def measure(self, func, *args):
        self.diffFromLast()
        self.resetMax()
        maxBefore = self.maximumValue()
        result = func(*args)
        usage = self.diffFromLast()
        maxUsage = self.maximumValue() - maxBefore

        return result, usage, maxUsage
    
    def snapshot(self):
        return torch.cuda.memory_snapshot()