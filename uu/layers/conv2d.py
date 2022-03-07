from typing import Dict
from torch import Tensor

import torch
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from  torch.nn.modules.conv import _ConvNd


class TiledConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        
        return input
    @staticmethod
    def backward(ctx, grad_output):
        return None


class TiledConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  
        is_ccheckpoint = False,
        name: str=""
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        self.is_ccheckpoint = is_ccheckpoint
        self.name = name
        super(TiledConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def set_cust_name(self, name):
        self.name = name
    def forward(self, *inputs) -> Tensor:
        # if type (inputs[0]) == tuple:
        #     # to remove additional packing in tuple
        #     inputs = list(inputs[0])
        # if len(inputs) == 2:
        #     input, info = inputs
        #     self.is_ccheckpoint = False
        # elif len(inputs) == 3:
        #     input, info, is_ccheckpoint = inputs
        #     self.is_ccheckpoint = is_ccheckpoint
        # else:
        #     print("missing info in cConv2d")
        #     assert False
        
        tconv2d = TiledConv2dFunction.apply
        print("cconv2D name:", self.name)
        # TODO, pass name to autofunction
        return tconv2d(inputs[0])