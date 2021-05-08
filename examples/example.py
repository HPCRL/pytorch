import math



import os
import torch
import argparse



import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from typing import List



from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init
import graphviz
from graphviz import Digraph

def make_dot(var, params):

    param_map = {id(v): k for k, v in params.items()}
    print(param_map)
    node_attr = dict(style='filled',
    shape='box',
    align='left',
    fontsize='12',
    ranksep='0.1',
    height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    seen = set()

def size_to_str(size):
    return '('+(', ').join(['%d'% v for v in size])+')'



# def add_nodes(var):

#     if var not in seen:

# if torch.is_tensor(var):

# dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')

# elif hasattr(var, 'variable'):

# u = var.variable

# node_name = '%s\n %s' % (param_map.get(id(u.data)), size_to_str(u.size()))

# dot.node(str(id(var)), node_name, fillcolor='lightblue')

# else:

# dot.node(str(id(var)), str(type(var).__name__))

# seen.add(var)

# if hasattr(var, 'next_functions'):

# for u in var.next_functions:

# if u[0] is not None:

# dot.edge(str(id(u[0])), str(id(var)))

# add_nodes(u[0])

# if hasattr(var, 'saved_tensors'):

# for t in var.saved_tensors:

# dot.edge(str(id(t)), str(id(var)))

# add_nodes(t)

# add_nodes(var.grad_fn)

# return dot

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
        print ("======================!!! weight size ", self.weight.size())
        if self.input_is_split:
            return F.linear(input, self.weight, self.bias)
        else:
            chunk_i = input.size()[0] // self.partition_degree_vect[0] #assume evenly divisible
            print("======================!!! chunk_i ", chunk_i)
            out_list = []

            for i in range(self.partition_degree_vect[0]):
                temp = torch.narrow(input, 0, i*chunk_i, chunk_i)
                print("======================!!! temp size ", temp.size())
                out_list.append(F.linear(temp, self.weight))
                print("======================!!! input size ", input.size())
        out = torch.cat(out_list, dim=0)
        print("======================!!! out size ", out.size())

        return out