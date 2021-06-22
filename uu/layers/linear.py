from .base_layer import BaseLayer
import torch.nn as nn
from torch import Tensor

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


    def mem_usage(self) -> int:
        print("in mem usage")


    # def check_compatibility() -> bool:

    
    # def hook(self, l: BaseLayer):
        
        



    def extra_repr(self) -> str:
        return 'uu.Linear id:{} [in_features={}, out_features={}, bias={}]'.format(
            self.unique_id, self.in_features, self.out_features, self.bias is not None
        )

 