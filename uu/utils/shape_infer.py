import torch
import torch.nn as nn
from uu.utils import correctness_check 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, relu
from torch.nn.parameter import Parameter
import math

class in_out_shape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape # tuple
        self.output_shape = output_shape # tuple
    def __repr__(self) -> str:
        rep = "input shape " + "".join([str(x)+"," for x in self.input_shape]) \
        + " -- output shape "+ "".join([str(x)+"," for x in self.output_shape])
        return rep 

#we only consider 4D tensor, later can extend to arbitrary
def shape_infer_sequence(seq_ops, inputH, inputW, N, C):
    H = inputH
    W = inputW
    shape_dict = {}
    #print("Input {}x{}x{}x{}".format(N, C, H, W))
    for op in seq_ops._modules.values():
        #print("L-->R current op", id(op))
        if isinstance(op, tilesplit.TiledSplit) or isinstance(op, relu.cReLu):
            continue
        if isinstance(op, conv2d.TiledConv2d):
            stride = op.stride
            pad = op.padding
            input_shape = (N, C, H, W)
            RS = op.kernel_size[0]
            C = op.out_channels
            H = math.floor((H+2*pad[0]-(RS-1)-1)/stride[0]+1)
            W = math.floor((W+2*pad[1]-(RS-1)-1)/stride[1]+1)
            output_shape = (N, C, H, W)
            in_out_shape_info = in_out_shape(input_shape, output_shape)
            shape_dict[id(op)] = in_out_shape_info

            #print("after conv2d {}x{}x{}x{}".format(N, C, H, W))
        elif isinstance(op, maxpool2d.cMaxPool2d):
            stride = op.stride
            pad = op.padding
            RS = op.kernel_size[0]
            input_shape = (N, C, H, W)
            H = math.floor((H+2*pad[0]-(RS-1)-1)/stride[0])+1
            W = math.floor((W+2*pad[1]-(RS-1)-1)/stride[1])+1
            output_shape = (N, C, H, W)
            in_out_shape_info = in_out_shape(input_shape, output_shape)
            shape_dict[id(op)] = in_out_shape_info
            #print("after maxpool2d {}x{}x{}x{}".format(N, C, H, W))
        else:
            input_shape = (N, C, H, W)
            output_shape = input_shape
            in_out_shape_info = in_out_shape(input_shape, output_shape)
            shape_dict[id(op)] = in_out_shape_info
            #print("here?? {}x{}x{}x{}".format(N, C, H, W))

    # give a hash-map to represnet input shape and out shape of a op
    return N, C, H, W, shape_dict