import torch
from uu.layers import base_layer 
from uu.utils import ftensor as ft

class BaseNetwork():
    def __init__(self):
        self.first_layer =  None
        #layer_id is only increased under one Network
        base_layer.BaseLayer.reset_glb_id()

    def check_compatibility(self, input: ft = None) -> bool:
        if self.first_layer is None:
            raise RuntimeError('Have to set first_layer before checking compatibility')
        if input is None:
            raise RuntimeError('Have to have a input tensor before checking compatibility')
        #TODO:
        
    def set_first_layer(self, l: base_layer.BaseLayer):
        self.first_layer = l

    def get_first_layer(self) -> base_layer.BaseLayer:
        return self.first_layer
    
    def mem_usage(self, input: ft = None) -> int:
        if self.first_layer is None:
            raise RuntimeError('Have to set first_layer before calculating mem usage')
        if input is None:
            raise RuntimeError('Have to have a input tensor before calculating mem usage')
    
    def travese(self):
        print(self.first_layer)

 