from typing import Dict
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


class TiledConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride,
                        padding, dilation, groups, info, uniq_id, is_ccheckpoint):
        print("== tiled conv2d forward")
        #print("input shape", input.size())
        c_info = info[0][uniq_id]   
        #print("current fwd info", c_info)
        s_depth = c_info.local_idex  # depth in current segment
        if c_info.op_idex == 0: # if it is the first conv in a segment then padding
            padding_info = c_info.padding_info
            pd = torch.nn.ConstantPad2d(padding_info, 0)
            input = pd(input)

        #print("af input shape", input.size())
        ctx.info = info           
        if s_depth == 0: 
            # depth is 1 if it is the last conv or the last one in segment
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
                    
            # TODO: how to save , stride, padding, dilation, groups ??
            #print("net_ out\n", out)
            # TODO : how to get the direct children after this??
            next_id = c_info.next_id
            input_tile_for_next = padding_calc.recreate_input_tile_f(info, out, next_id)
            # print("input_tile_for_next\n", input_tile_for_next)
            out = input_tile_for_next
            #print("shape input_tile_for_next\n", input_tile_for_next.size())

        return out

    #@staticmethod
    #def backward(ctx, grad_output):
    #   return
        # print("\n** tiled conv2d backward")
        # # print("** grad_output", grad_output)
        # # print("** grad_output shape", grad_output.size())
        # depth = ctx.depth *-1
        # print("ctx.depth", ctx.depth)
        # print("depth", depth)
        # info = ctx.info
        # print("info coord", info[depth].coord)
        # input = ctx.saved_tensors[0]
        # weight = ctx.weight
        # #print("in_ grad_out shape", grad_output)
        # #print("input shape", input.size())
        # #print("weight shape", weight.size())
        # grad_input = None
        # grad_weight = None
        # c_info = info[depth]
        # ordering = c_info.ordering_info
        # if ctx.needs_input_grad[0]:
        #     if depth == ctx.num_conv * -1:
        #         # for user input
        #         print("AA")
        #         weight = Parameter(torch.rot90(weight.data, 2, [2,3])).transpose(0,1)
        #         # print("input shape", input.size())
        #         # print("out shape", out.size())
        #         # print("weight shape", weight.size())
        #         # print("grad_output shape", grad_output.size())
        #         grad_input = F.conv2d(grad_output, weight)
        #         print("final", grad_input.size(), grad_input)
        #     elif depth == -1:
        #         print("AAA")
        #         # a whole grad_output as input of backward
        #         new_grad_out = padding_calc.get_input_tile(info, grad_output, depth)
        #         # since I remove padding from get_input_tile, so manually do it here.
        #         padding_info = c_info.padding_info
        #         pd = torch.nn.ConstantPad2d(padding_info, 0)
        #         new_grad_out = pd(new_grad_out)

        #         print("new_grad_out", new_grad_out.size())
        #         print(padding_info)
        #         weight = Parameter(torch.rot90(weight.data, 2, [2,3])).transpose(0,1)
        #         grad_input = F.conv2d(new_grad_out, weight)
        #         input_tile_for_next = padding_calc.recreate_input_tile(ctx.info, grad_input, depth-1)
        #         grad_input = input_tile_for_next
        #         print("grad_input", grad_input.size())
        #     elif ordering[2] == 0:  
        #         print("S_last")
        #         # the last conv in local continous conv segment
        #         padding_info = c_info.padding_info
        #         pd = torch.nn.ConstantPad2d(padding_info, 0)
        #         new_grad_out = pd(grad_output)

        #         print("new_grad_out", new_grad_out.size())
        #         # print(new_grad_out)
        #         weight = Parameter(torch.rot90(weight.data, 2, [2,3])).transpose(0,1)
        #         grad_input = F.conv2d(new_grad_out, weight)
        #         input_tile_for_next = padding_calc.recreate_input_tile(ctx.info, grad_input, depth-1)
        #         grad_input = input_tile_for_next
        #         print("grad_input", grad_input.size())
        #     else:
        #         print("MAA")
        #         weight = Parameter(torch.rot90(weight.data, 2, [2,3])).transpose(0,1)
        #         print("new_grad_out", grad_output.size())
        #         grad_input = F.conv2d(grad_output, weight)
        #         input_tile_for_next = padding_calc.recreate_input_tile(ctx.info, grad_input, depth-1)
        #         grad_input = input_tile_for_next
        #         print("grad_input", grad_input.size())
                

        # if ctx.needs_input_grad[1]:
        #     if depth == ctx.num_conv-1:
        #         #print("info", info)
        #         #print("BB")
        #         Th = info[depth].pt_size[2]
        #         Tw = info[depth].pt_size[3]
        #         H_index = info[depth].coord[0]* Th
        #         W_index = info[depth].coord[1]* Tw
        #         new_grad_out = grad_output[:,:, H_index:H_index+Th, W_index:W_index+Tw]
        #         grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, new_grad_out)
        #         #grad_weight = torch.zeros(grad_weight.shape).cuda()
        #         #print("grad_weight", grad_weight.size())
        #     else:
        #         #need to get the correct tile 
        #         H_len = grad_output.size()[2]
        #         W_len = grad_output.size()[3]
        #         # print("BBB")
        #         # print("depth", depth)
        #         # print("compute grad_weight\n")
        #         # print("input shape", input.size())
        #         depth += 1  # TODO: rethink about it
        #         new_grad_out = grad_output[:,:, depth:H_len-depth, depth:W_len-depth]
        #         input_H = input.size()[2]
        #         input_W = input.size()[3]
        #         #TODO: here needs change !!!
        #         new_input = input[:,:, ctx.depth:input_H-ctx.depth , ctx.depth:input_W-ctx.depth]
        #         #print("$$$$$$$$$$$$$$$$$$$$$grad_out {} \n new_grad_out {}".format(grad_output, new_grad_out))
        #         # print("compute grad_weight\n")
        #         # print("input shape", input.size())
        #         # print("new_input shape", new_input.size())
        #         # print("grad_out shape", grad_output.size())
        #         # print("new_grad_out shape", new_grad_out.size())
        #         #grad_weight = torch.zeros(weight.shape).cuda()
        #         grad_weight = torch.nn.grad.conv2d_weight(new_input, weight.shape, new_grad_out)
        #         #grad_weight = torch.zeros(grad_weight.shape).cuda()
        #         #print("grad_weight", grad_weight.size())

                
        # grad_bias = None #TODO: bias shape??
        # return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None

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
        # depth: int =  0,
        # num_conv: int = 0,  # num of conv in segment
        is_ccheckpoint = False
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        # self.depth = depth
        # self.num_conv = num_conv
        self.is_ccheckpoint = is_ccheckpoint
        super(TiledConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, *inputs) -> Tensor:
        print("id", id(self))
        first_op = False
        if type (inputs[0]) == tuple:
            # to remove additional packing in tuple
            inputs = list(inputs[0])
        if len(inputs) == 2:
            input, info = inputs
            self.is_ccheckpoint = False
        elif len(inputs) == 3:
            input, info, is_ccheckpoint = inputs
            self.is_ccheckpoint = is_ccheckpoint
        else:
            print("missing info in cConv2d")
            assert False
        
        # TODO: I need to pass hash val to fwd and bwd
        tconv2d = TiledConv2dFunction.apply
        uniq_id = id(self)
        pi = info[0][uniq_id]
        self.padding = (0,0) #force no auto padding in our customized functions.
        if pi.op_idex == 0:
           return tconv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups, info, uniq_id, self.is_ccheckpoint)
        else:
           return tconv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups, info, uniq_id, self.is_ccheckpoint), info, self.is_ccheckpoint


                
 