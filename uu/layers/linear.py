from .base_layer import BaseLayer
import torch.nn as nn
from torch import Tensor
from uu.utils import ftensor as ft
import numpy as np

print("imported")

class Linear(BaseLayer):
    in_features: int
    out_features: int
    bias: bool
    weight: Tensor
    op: 'Module'
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if __debug__:
            print("linear init")
            print("my_id", self.unique_id)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.op = None

  
    def forward(self, input: Tensor) -> Tensor:
        if __debug__:
            print("in linear forward")
            print("input", input)
        self.op = nn.Linear(self.in_features, self.out_features, self.bias)
        if input.get_device() >= 0:
            self.op.to(input.get_device())
        return self.op(input)


    def mem_usage(self, input: ft) -> int:
        w_mu = (self.in_features*self.out_features)
        if self.bias:
            b_mu = self.out_features
        out_shape = input.get_dims()

        out_shape[len(out_shape)-1] = self.out_features
        output = ft.FakeTensor(out_shape)
        del out_shape       #delete temp list
        o_mu = output.size()

        if __debug__:
            print("in linear mem usage")
            print("weight size [{} x {}] = {}".format(self.in_features, self.out_features, w_mu))
            print("output size {} = {}".format(output, o_mu))
            if self.bias: 
                print("bias size [{} x {}] = {}".format(self.in_features, self.out_features, b_mu))
            print("** cur op total size ", (w_mu+o_mu+b_mu))

        return (w_mu+o_mu+b_mu), output





    # def check_compatibility() -> bool:

    
    # def hook(self, l: BaseLayer):
        
        
    def extra_repr(self) -> str:
        return 'uu.Linear id:{} [in_features={}, out_features={}, bias={}]'.format(
            self.unique_id, self.in_features, self.out_features, self.bias is not None
        )

 