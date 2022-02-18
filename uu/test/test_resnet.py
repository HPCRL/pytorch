import os
import torch
import argparse

import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from typing import List

from uu.utils.construct_autograph import Graph, build_graph

from functools import partial

# class Conv2dAuto(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
# conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False) 

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.in_channels, self.out_channels =  in_channels, out_channels
#         self.blocks = nn.Identity()
#         self.shortcut = nn.Identity()   
    
#     def forward(self, x):
#         residual = x
#         if self.should_apply_shortcut: residual = self.shortcut(x)
#         x = self.blocks(x)

#         print("x size ", x.size())
#         print("residual size ", residual.size())

#         x += residual
#         return x
    
#     @property
#     def should_apply_shortcut(self):
#         return self.in_channels != self.out_channels


# from collections import OrderedDict

# class ResNetResidualBlock(ResidualBlock):
#     def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
#         super().__init__(in_channels, out_channels)
#         self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
#         self.shortcut = nn.Sequential(OrderedDict(
#         {
#             'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
#                       stride=self.downsampling, bias=False),
#             'bn' : nn.BatchNorm2d(self.expanded_channels)
#         })) if self.should_apply_shortcut else None
        
        
#     @property
#     def expanded_channels(self):
#         return self.out_channels * self.expansion
    
#     @property
#     def should_apply_shortcut(self):
#         return self.in_channels != self.expanded_channels


# from collections import OrderedDict
# def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
#     return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
#                           'bn': nn.BatchNorm2d(out_channels) }))

# class ResNetBasicBlock(ResNetResidualBlock):
#     expansion = 1
#     def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
#         super().__init__(in_channels, out_channels, *args, **kwargs)
#         self.blocks = nn.Sequential(
#             conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
#             activation(),
#             conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
#         )
    
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



def dryrun_cpu(model: nn.Module, input_dim: List[int]) -> None:
    device_cpu = torch.device("cpu")
    model.to(device_cpu)
    input = torch.rand(input_dim).cpu()
    print("input tensor size", input.size())
    g = build_graph(model, input, True)
    g.save("test_resnet.pdf")





    

def main():
    dummy = torch.ones((1, 64, 14, 14))
    block = BasicBlock(64, 64)
    print(block)


    dryrun_cpu(block, [1, 64, 14, 14])




    #out = block(dummy)

    #print("out size ", out.size())


       
if __name__=="__main__":

    main()