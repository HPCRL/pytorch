from typing import List
import numpy as np

class FakeTensor():
    def __init__(self, dims: List[int]=[]):
        if len(dims) <= 0:
            raise TypeError('FakeTensor must have at least one dimension')
        self.dims = dims
    
    def size(self) -> int:
        return np.prod(self.dims)

