import os
import torch
import argparse

import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from typing import List
import operator
from uu.utils.make_graph import make_graph, save_graph
from functools import partial
import uu
from uu.layers import relu

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

from torch.fx import symbolic_trace
from torch.fx.passes.graph_drawer import FxGraphDrawer
import pydot

def main():
    block = Net()
    print(block)
    traced = torch.fx.symbolic_trace(block)
    print(traced.code)


    patterns = set([ "block1.relu"])
    # Go through all the nodes in the Graph
    for n in traced.graph.nodes:
        # If the target matches one of the patterns
        print("n.target", n.target)
        if any(n.target == pattern for pattern in patterns):
            # Set the insert point, add the new node, and replace all uses
            # of `n` with the new node
            print("replaced??")
            with traced.graph.inserting_after(n):
                new_node = traced.graph.call_function(relu.cReLu, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            # Remove the old node from the graph
            traced.graph.erase_node(n)

    # Don't forget to recompile!
    traced.recompile()
    print(traced.code)

     # Convert the fx graph to an internal graph representation
    name_to_module = {}
    for name, module in block.named_modules():
        name_to_module[name] = module
    graph = []
    for fx_node in traced.graph.nodes:
        if fx_node.op == "call_module":
            module_name = fx_node.target
            module = name_to_module[module_name]
            print("module_name", module_name)
        elif fx_node.op == "placeholder":
            print("placeholder", fx_node)
        elif fx_node.op == "get_attr":
            print("get_attr", fx_node)
        elif fx_node.op == "call_function" or fx_node.op == "call_method":
            print("call f/m", fx_node)
        elif fx_node.op == "output":
            print("output", fx_node)
        else:
            assert 0, f"Unknown operator type: {fx_node.op}"
        
    drawer = FxGraphDrawer(
                traced, "preview", ignore_getattr=True
            )
    dot_graphs = drawer.get_all_dot_graphs()
    print("type ", type(dot_graphs))
    for name, dot_graph in dot_graphs.items():
        print("name ", name)
        dot_graph.write_raw(f"{name}.dot")

    

    (graph,) = pydot.graph_from_dot_file('preview.dot')
    graph.write_png('preview.png')
       
if __name__=="__main__":

    main()