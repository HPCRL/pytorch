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

        total_mem = self.mem_collect(input, self.first_layer)
        print("entwork total mem", total_mem)

    
    def mem_collect(self, input: ft, l: base_layer.BaseLayer):
        if l is None:
            return 0
        l.fwdcounter_inc()
        # join node, not fullly arrived.
        if l.fwdcounter < l.get_prev_layer_size():
            return 0
    
        mem_u, output = l.mem_usage(input) 
        next_layers = l.get_next()
        if len(next_layers) == 1:
            mem_u += self.mem_collect(output, next_layers[0])
        elif len(next_layers) > 1:
            for i in range(0, len(next_layers)):
                mem_u += self.mem_collect(output, next_layers[i])
        
        l.reset_fwdcounter()
        return mem_u
    

    
    #DFS
    def traverse(self, l: base_layer.BaseLayer):
        if l is None:
            return
        l.fwdcounter_inc()
        # join node, not fullly arrived.
        if l.fwdcounter < l.get_prev_layer_size():
            return

        print("trv ", l)
        next_layers = l.get_next()
        if len(next_layers) == 1:
            self.traverse(next_layers[0])
        elif len(next_layers) > 1:
            for i in range(0, len(next_layers)):
                self.traverse(next_layers[i])
        
        l.reset_fwdcounter()
    



        
        

 