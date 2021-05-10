
import math
import functools

import os
import torch
import argparse

import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from typing import List
from torch.autograd import Variable

from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn  import init

class TiledLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, input_is_split = False, partition_degree_vect=[1,1,1]):
        ctx.save_for_backward(input, weight, bias)
        ctx.loops = partition_degree_vect[0]
        ctx.input_is_split = input_is_split
        if input_is_split:
            return F.linear(input, weight, bias)
        else:
            chunk_i = input.size()[0] // partition_degree_vect[0]   #assume evenly divisible
            out_list = []
            in_list = []
            for i in range(partition_degree_vect[0]):
                temp = Variable(torch.narrow(input, 0, i*chunk_i, chunk_i).data, requires_grad=True)
                in_list.append(temp)
                tweight = Variable(weight.data, requires_grad=True)
                with torch.set_grad_enabled(True):
                    res = F.linear(temp, weight)
                out_list.append(res)
        
            out = torch.cat(out_list, dim=0)
            ctx.outlist = out_list
            ctx.inlist = in_list
            ctx.chunk_i = chunk_i
            return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = None
        gin = gw = None
        
        grad_input_list = []
        grad_weight_list = []
        input, weight, bias = ctx.saved_tensors
        output_list = ctx.outlist
        input_list = ctx.inlist

    
        if ctx.input_is_split:
            return 0 #Todo: 
        else:
            print("======================!!! ctx", ctx.loops)
            chunk_i = ctx.chunk_i
            for i in range(ctx.loops):
                out_slice = output_list[i]
                inp_slice = input_list[i]
                grad_slice = torch.narrow(grad_output, 0, i*chunk_i, chunk_i)

                partial_grad_in = torch.autograd.grad(out_slice, inp_slice, grad_slice, retain_graph =True)
                grad_input_list.append(partial_grad_in[0])

                partial_grad_w = torch.autograd.grad(out_slice, weight, grad_slice, retain_graph =True)
                grad_weight_list.append(partial_grad_w[0])

            print("grad_in_list", grad_input_list)
            gin = torch.cat(grad_input_list, dim=0)
            gw = grad_weight_list[0]
            #print(grad_weight_list)
            for i in range(1, ctx.loops):
                gw += grad_weight_list[i]
            
            print("gin ", gin)
            print("gw ", gw)
            # print("grad_output", grad_output)
            # print("weight", weight)
        
            #Manual reference.
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)

            print("grad_input ", grad_input)
            print("grad_weight ", grad_weight)
            # if bias is not None and ctx.needs_input_grad[2]:
            #     grad_bias = grad_output.sum(0)

            return grad_input, grad_weight, grad_bias, None, None


class TiledLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    partition_degree_vect: List[int] #[i,j,k]
    weight: Tensor
    input_is_split: bool

    def __init__(self, in_features: int, out_features: int, bias: bool = False, partition_degree_vect = [1,1,1], input_is_split = False) -> None:
        super(TiledLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.reshape(torch.arange(0, self.out_features* self.in_features, step=1.0, dtype=torch.float), (self.out_features, self.in_features))) # J, K
        self.input_is_split = input_is_split
        self.partition_degree_vect = partition_degree_vect
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        #print ("weight ", self.weight)
        return TiledLinearFunction.apply(input, self.weight, self.bias, self.input_is_split, self.partition_degree_vect)
        

class Net(nn.Module):
    def __init__(self, b, m, n, k) :
        super(Net, self).__init__()
        self.linear_1 = TiledLinear(k,n, partition_degree_vect = [4,1,1])    #reduce, out
        
    def forward(self, x):
        result_1 = self.linear_1(x)
        return result_1



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = 1
    m = 8
    n = 6
    k = 4

    model = Net(b, m, n, k)
    model.to(device)
    input = torch.rand(m, k).cuda()
    input.requires_grad=True
    output = model(input)
    erro = output
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", erro)
    erro.backward(torch.ones_like(erro))




   
if __name__=="__main__":
    main()

