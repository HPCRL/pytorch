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
from uu.layers import conv2d
from torch.nn.parameter import Parameter
import numpy as np


"""Network define START()"""
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv = conv2d.TiledConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
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
        self.block2 = BasicBlock(64, 64)

    def forward(self, x):
        """Network Forward START()"""
        out = self.block1(x)
        out = self.block2(out)
        return out
        """Network Forward END()"""

"""Network define END()"""


from torch.fx.passes.graph_drawer import FxGraphDrawer
import pydot
from uu.utils.gen_SFT_code import rewriteCode
from uu.utils.binding_name import binding_module_name
def main():
    network = Net()
    print("network", network)
    binding_module_name(network)
    


if __name__=="__main__":

    main()