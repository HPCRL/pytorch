from __future__ import annotations
from typing import List
import torch

class BaseLayer(torch.nn.Module):
    glb_id : int = 0
    unique_id : int
    next_layers: List['BaseLayer']
    prev_layers: List['BaseLayer']

    def __init__(self):
        if __debug__:
            print("baselayer init")
        super(BaseLayer, self).__init__()
        self.next_layers = list()
        self.prev_layers = list()
        BaseLayer.glb_id += 1
        self.unique_id = BaseLayer.glb_id 
        self.fwdcounter = 0
        self.bwdcounter = 0
        
    def forward(self, x):
        pass

    def extra_repr(self) -> str:
        return 'uu.BaseLayer'

    def get_next(self) -> List['BaseLayer']:
        if __debug__:
            self.next_layers.sort(key=lambda x:x.unique_id)
        return self.next_layers
    
    def get_next_layer_size(self) -> int:
        return len(self.next_layers)

    def get_prev(self) -> List['BaseLayer']:
        if __debug__:
            self.prev_layers.sort(key=lambda x:x.unique_id)
        return self.prev_layers
    
    def get_prev_layer_size(self) -> int:
        return len(self.prev_layers)

    def hook(self, l: BaseLayer):
        if __debug__:
            print("hook layer {}:{} <-> {}:{}\n".format(self.unique_id, self.extra_repr(), l.unique_id, l.extra_repr()) )
        self.next_layers.append(l)
        l.prev_layers.append(self)

    def mem_usage(self) -> int:
        return 0
    
    def reset_glb_id():
        BaseLayer.glb_id = 0

    def fwdcounter_inc(self):
        self.fwdcounter += 1
    
    def bwdcounter_inc(self):
        self.bwdcounter += 1
    
    def reset_fwdcounter(self):
        self.fwdcounter = 0
    
    def reset_bwdcounter(self):
        self.bwdcounter = 0
    

    


    


