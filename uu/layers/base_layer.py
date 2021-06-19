from __future__ import annotations
from typing import List
import torch

class BaseLayer(torch.nn.Module):
    glb_id : int = 0
    unique_id : int
    next_layers: List['BaseLayer']
    prev_layers: List['BaseLayer']

    def __init__(self):
        print("baselayer init")
        super(BaseLayer, self).__init__()
        self.next_layers = list()
        self.prev_layers = list()
        BaseLayer.glb_id += 1
        self.unique_id = BaseLayer.glb_id 
        
    def forward(self, x):
        pass

    def extra_repr(self) -> str:
        return 'uu virtual baselayer'

