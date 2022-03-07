import os
import torch
import argparse

import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from uu.utils.correctness_check import check_equal 

from typing import List
import operator
from uu.utils.internal_graph import create_internal_graph
from functools import partial
import uu
from uu.layers import relu
from torch.nn.parameter import Parameter
import numpy as np

"""Network define START()"""
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    
    print("$$---", conv.extra_repr())
    val = np.arange(in_planes*out_planes*3*3)
    w1 = np.reshape(val, (out_planes, in_planes, conv.kernel_size[0], conv.kernel_size[1]))
    conv.weight = Parameter(torch.Tensor(w1))
    return conv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = BasicBlock(64, 64)
        self.mxp = nn.MaxPool2d((2,2), (2,2))
        self.block2 = BasicBlock(64, 64)
    
    def forward(self, x): 
        """Network Forward START()"""  
        out = self.block1(x)
        out = self.mxp(out)
        out = self.block2(out)
        return out
        """Network Forward END()"""  

"""Network define END()"""

class test_Net(nn.Module):
    def __init__(self):
        super(test_Net, self).__init__()
        self.block1 = BasicBlock(64, 64)
        self.mxp = nn.MaxPool2d((2,2), (2,2))
        self.block2 = BasicBlock(64, 64)
    

    def forward(self, x):
        block1_conv1 = self.block1.conv1(x);  x = None
        block1_conv2 = self.block1.conv2(block1_conv1);  block1_conv1 = None
        mxp = self.mxp(block1_conv2);  block1_conv2 = None
        block2_conv1 = self.block2.conv1(mxp);  mxp = None
        block2_conv2 = self.block2.conv2(block2_conv1);  block2_conv1 = None
        return block2_conv2
        
from torch.fx import symbolic_trace
from torch.fx.passes.graph_drawer import FxGraphDrawer
import pydot
from uu.utils.gen_SFT_code import rewriteCode

def main():
    network = Net()
    print("network", network)
    
    traced = symbolic_trace(network)
    print(traced.graph)
    print(traced.code)

    input = torch.rand(1,64,7,7)

    outref = network(input)

    test_net = test_Net()
    outtest = test_net(input)

    check_equal(outref, outtest, False)



    input_shape = (1, 3, 7, 7)
    create_internal_graph(network, [input_shape], 0, True)


    rewriteCode("test_iG.py")
  
    

    


       
if __name__=="__main__":

    main()