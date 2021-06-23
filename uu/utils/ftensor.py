from typing import List
import numpy as np

class FakeTensor():
    def __init__(self, dims: List[int]=[]):
        if len(dims) <= 0:
            raise TypeError('FakeTensor must have at least one dimension')
        self.dims = dims
    
    def size(self) -> int:
        return np.prod(self.dims)
    
    def get_dims(self) -> List[int]:
        return self.dims
    
    def __repr__(self) -> str:
        s = ""
        for i in self.dims:
            s = s + str(i) + " x "
        s = s[:-3]
        return 'fTensor [{}]'.format(s)

