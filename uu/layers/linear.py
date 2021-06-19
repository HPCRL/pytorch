from .base_layer import BaseLayer
import torch.nn as nn
from torch import Tensor

class Linear(BaseLayer):
    in_features: int
    out_features: int
    bias: bool
    weight: Tensor
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        print("linear init")
        super().__init__()
        print("my_id", self.unique_id)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
    def forward(self, input: Tensor) -> Tensor:
        print("in linear forward")
        print("input", input)

        if input.get_device() >= 0:
            l = nn.Linear(self.in_features, self.out_features, self.bias).to(input.get_device())
        else:
            l = nn.Linear(self.in_features, self.out_features, self.bias)
        return l(input)


    def extra_repr(self) -> str:
        return 'uu in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

 