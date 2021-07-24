from typing import Dict, List, Union, Tuple, Optional
from torch import Tensor
import torch
from torch.nn import functional as F

class TiledConcatenateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *input):
        dim = input[-1]
        input = input[:-1]
        # print ("**input[0]", input[0])
        # print ("dim", dim)
        output = torch.cat(tensors=input, dim=dim, out=None)
        #output.requires_grad = True #tensors[0].requires_grad
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        #print("\n^^^^^grad_output", grad_output, grad_output.size())
        return grad_output, grad_output, grad_output, None


# class TiledCat(torch.nn.Module):
#     def __init__(self):
#         super(TiledCat, self).__init__()
#         self.activation_post_process = torch.nn.Identity()

#     def forward(self, tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: int=0, out: Optional[Tensor]=None):
#         print("here")
#         tcat = TiledConcatenateFunction.apply
#         r = tcat(tensors, dim, out)
#         return r
 
