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
from torch.nn.parameter import Parameter
from torch.autograd.variable import Variable


class TiledConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride,
                        padding, dilation, groups, info, depth, num_conv, is_ccheckpoint):
        print("== tiled conv2d forward")
        print("depth", depth)
        # print("is_ccheckpoint", is_ccheckpoint)

        ctx.depth = depth
        ctx.info = info   
        ctx.num_conv = num_conv             
        if depth == 0:
            if not is_ccheckpoint:    
                ctx.save_for_backward(input)
                ctx.weight = weight
                out = F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
            else:
                out = F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
        else:
            if not is_ccheckpoint:            
                ctx.save_for_backward(input)
                ctx.weight = weight
                out = F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
            else:
                out = F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
                    
            # how to save , stride, padding, dilation, groups ??
            #print("net_ out\n", out)
            input_tile_for_next = padding_calc.recreate_input_tile(info, out, depth-1)
            #print("shape input_tile_for_next\n", input_tile_for_next.size())
            #print("input_tile_for_next\n", input_tile_for_next)
            out = input_tile_for_next
         
        return out

    @staticmethod
    def backward(ctx, grad_output):
        print("** tiled conv2d backward")
        # print("** grad_output", grad_output)
        # print("** grad_output shape", grad_output.size())
        print("ctx.depth", ctx.depth)
        depth = ctx.num_conv-ctx.depth-1
        info = ctx.info
        print("info coord", info[depth].coord)
        input = ctx.saved_tensors[0]
        weight = ctx.weight
        # print("in_ grad_out shape", grad_output)

        # print("input shape", input)
        print("weight shape", weight.size())
        grad_input = None
        grad_weight = None
        if ctx.needs_input_grad[0]:
            if depth == 0:
                # for user input
                #print("AA")
                weight = Parameter(torch.rot90(weight.data, 2, [2,3])).transpose(0,1)
                # print("input shape", input.size())
                # print("out shape", out.size())
                # print("weight shape", weight.size())
                # print("grad_output shape", grad_output.size())
                grad_input = F.conv2d(grad_output, weight)
                #print("final", grad_input.size())
            elif depth == ctx.num_conv-1:
                #print("AAA")
                # a whole grad_output as input of backward
                new_grad_out = padding_calc.get_input_tile(info, grad_output, depth)
                weight = Parameter(torch.rot90(weight.data, 2, [2,3])).transpose(0,1)
                grad_input = F.conv2d(new_grad_out, weight)
                input_tile_for_next = padding_calc.recreate_input_tile(ctx.info, grad_input, depth-1)
                grad_input = input_tile_for_next
                #print(grad_input)
            else:
                #print("AAAA")
                weight = Parameter(torch.rot90(weight.data, 2, [2,3])).transpose(0,1)
                grad_input = F.conv2d(grad_output, weight)
                input_tile_for_next = padding_calc.recreate_input_tile(ctx.info, grad_input, depth-1)
                grad_input = input_tile_for_next

        if ctx.needs_input_grad[1]:
            if depth == ctx.num_conv-1:
                #print("info", info)
                #print("BB")
                Th = info[depth].orig_size[2]
                Tw = info[depth].orig_size[3]
                H_index = info[depth].coord[0]* Th
                W_index = info[depth].coord[1]* Tw
                new_grad_out = grad_output[:,:, H_index:H_index+Th, W_index:W_index+Tw]
                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, new_grad_out)
                #grad_weight = torch.zeros(grad_weight.shape).cuda()
                #print("grad_weight", grad_weight.size())
            else:
                #need to get the correct tile 
                H_len = grad_output.size()[2]
                W_len = grad_output.size()[3]
                # print("BBB")
                # print("depth", depth)
                # print("compute grad_weight\n")
                # print("input shape", input.size())
                depth += 1  # TODO: rethink about it
                new_grad_out = grad_output[:,:, depth:H_len-depth, depth:W_len-depth]
                input_H = input.size()[2]
                input_W = input.size()[3]
                new_input = input[:,:, ctx.depth:input_H-ctx.depth , ctx.depth:input_W-ctx.depth]
                #print("$$$$$$$$$$$$$$$$$$$$$grad_out {} \n new_grad_out {}".format(grad_output, new_grad_out))
                # print("compute grad_weight\n")
                # print("input shape", input.size())
                # print("new_input shape", new_input.size())
                # print("grad_out shape", grad_output.size())
                # print("new_grad_out shape", new_grad_out.size())
                #grad_weight = torch.zeros(weight.shape).cuda()
                grad_weight = torch.nn.grad.conv2d_weight(new_input, weight.shape, new_grad_out)
                #grad_weight = torch.zeros(grad_weight.shape).cuda()
                #print("grad_weight", grad_weight.size())

                
        grad_bias = None #TODO: bias shape??
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None

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
        depth: int =  0,
        num_conv: int = 0,  # num of conv in segment
        is_ccheckpoint = False
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        self.depth = depth
        self.num_conv = num_conv
        self.is_ccheckpoint = is_ccheckpoint
        super(TiledConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, *inputs) -> Tensor:
        if len(inputs) == 2:
            input, info = inputs
            self.is_ccheckpoint = False
        else:
            input, info, is_ccheckpoint = inputs
            self.is_ccheckpoint = is_ccheckpoint
        tconv2d = TiledConv2dFunction.apply
        # return tconv2d(input, self.weight, self.bias, self.stride,
        #                 self.padding, self.dilation, self.groups, info, self.depth, self.num_conv), info


        if self.depth == 0:
           return tconv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups, info, self.depth, self.num_conv, self.is_ccheckpoint)
        else:
           return tconv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups, info, self.depth, self.num_conv, self.is_ccheckpoint), info, self.is_ccheckpoint


                
 