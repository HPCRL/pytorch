import torch
import torch.nn as nn
from uu.utils import correctness_check 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy
from torch.nn.parameter import Parameter
import math

#we only consider 4D tensor, later can extend to arbitrary
def shape_infer_sequence(seq_ops, inputH, inputW, C, N):
    H = inputH
    W = inputW
    print("Input {}x{}x{}x{}".format(N, C, H, W))
    for op in seq_ops._modules.values():
        if isinstance(op, conv2d.TiledConv2d):
            stride = op.stride
            pad = op.padding
            RS = op.kernel_size[0]
            C = op.out_channels
            H = math.floor((H+2*pad[0]-(RS-1)-1)/stride[0])+1
            W = math.floor((W+2*pad[1]-(RS-1)-1)/stride[1])+1

            print("after conv2d {}x{}x{}x{}".format(N, C, H, W))
        if isinstance(op, maxpool2d.cMaxPool2d):
            stride = op.stride
            pad = op.padding
            RS = op.kernel_size[0]
            H = math.floor((H+2*pad[0]-(RS-1)-1)/stride[0])+1
            W = math.floor((W+2*pad[1]-(RS-1)-1)/stride[1])+1
            print("after maxpool2d {}x{}x{}x{}".format(N, C, H, W))

    return N, C, H, W