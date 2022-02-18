import os
import torch
import argparse

import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from typing import List

from uu.utils.make_graph import make_graph, save_graph
from functools import partial


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = BasicBlock(64, 64)
        self.mxp = nn.MaxPool2d((2,2), (2,2))
        self.block2 = BasicBlock(64, 64)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.mxp(out)
        out = self.block2(out)

        return out



def main():
    dummy = torch.ones((1, 64, 14, 14))
    block = Net()
    print(block)
    # we make a scripted function
    fn = torch.jit.trace(block,dummy)
    trace_graph = fn.graph_for(dummy)

    print(type(fn), type(fn.code), type(fn.graph_for), type(fn.graph))

    print(dir(torch._C.Node))

    print("==============================================")
    print(fn.code, fn.graph_for, fn.graph)
    print("==============================================")
    g = make_graph(trace_graph)
   

    save_graph(g, "test_resnet1.pdf")
  

       
if __name__=="__main__":

    main()