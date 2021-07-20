from typing import Dict
from .base_layer import BaseLayer
import torch.nn as nn
from torch import Tensor
from uu.utils import ftensor as ft
import numpy as np
import torch
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from  torch.nn.modules.conv import _ConvNd
from torch.nn import functional as F
from uu.utils import padding_calc

class TiledConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride,
                        padding, dilation, groups, info, depth):
        if depth == 0:
            out = F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
        else:
            out = F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
            
            #print("net_ out\n", out)
            input_tile_for_next = padding_calc.recreate_input_tile(info, out, depth-1)
            #print("shape input_tile_for_next\n", input_tile_for_next.size())
            #print("input_tile_for_next\n", input_tile_for_next)
            out = input_tile_for_next
         
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.zeros([2, 4])
        return grad_input

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
        # additional info
        depth: int =  0
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        self.depth = depth
        super(TiledConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input: Tensor, info: Dict) -> Tensor:
        tconv2d = TiledConv2dFunction.apply
        return tconv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups, info, self.depth)
 