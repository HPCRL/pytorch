from .base_layer import BaseLayer
import torch.nn as nn
from torch import Tensor

class Linear(BaseLayer):
    in_features: int
    out_features: int
    bias: bool
    weight: Tensor
    


    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        print("my_id", self.unique_id)

#     def forward(self, input: Tensor) -> Tensor:
#         return nn.Linear(in_features, out_features, bias)


 