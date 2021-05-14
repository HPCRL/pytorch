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
from torch.nn  import init



class TiledConv2D(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups' , 'padding_mode'
                        'output_padding' , 'in_channels' , 'out_channels', 'kernel_size']
    
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    groups: int
    bias: Tensor
    partition_degree_vect: List[int]
    weight: Tensor
    input_is_split: bool

    def __init__(self, in_channels: int,  out_channels: int, kernel_size: int, stride=1, padding=0,
                                 dilation=1, groups=1, bias=None, partition_degree_vect=[1,1,1,1], input_is_split=False) -> None:
        
        super(TiledConv2D, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.reshape(torch.arange(0, self.out_channels * self.in_channels
                                                                   * self.kernel_size* self.kernel_size, step=1.0, dtype=torch.float), (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size )))
        self.stride = stride    #Stride of convolution Default 1
        self.padding = padding  #Zero Padding added to both sides of the input. Default 0
        self.dilation = dilation #Spacing between Kernel element
        self.groups = groups
        self.bias = bias

        self.input_is_split = input_is_split
        self.partition_degree_vect = partition_degree_vect

        # if bias:
        #     self.bias = Parameter(torch.empty(kernel_size))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()
        


    def reset_parameters(self) -> None:
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    

    def forward(self, input: Tensor) -> Tensor:

        print("++++++++++++++++++++++" )
        if self.input_is_split:
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        else:
            chunk_i_x = input.size()[2] // self.partition_degree_vect[0]  # 2D square tile  - Number of tiles in x
            chunk_i_y = input.size()[3] // self.partition_degree_vect[1]  # 2D square tile  - Number of tiles in y

            print("======================!!! chunk_i_x ", chunk_i_x)
            print("======================!!! chunk_i_y ", chunk_i_y)

            out_list = []

            for i in range(self.partition_degree_vect[0]):

                out_list_row = []
                
                for j in range(self.partition_degree_vect[1]):
                    if i == 0 and j == 0 :
                        temp = torch.narrow(input, 2,  i*chunk_i_y, chunk_i_y +1)
                        temp2 = torch.narrow(temp, 3,  j*chunk_i_x, chunk_i_x +1)
                        print("======================!!! temp size ", temp.size())
                        print("======================!!! temp2 size ", temp2.size())
                       
                    elif i == 0 and j != 0 and j != self.partition_degree_vect[1] -1 :
                        temp = torch.narrow(input, 2,  i*chunk_i_y, chunk_i_y +1)
                        temp2 = torch.narrow(temp, 3,  j*chunk_i_x -1 , chunk_i_x +2)
                        print("======================!!! temp size ", temp.size())
                        print("======================!!! temp2 size ", temp2.size())
                       
                    elif i == 0 and j == self.partition_degree_vect[1] -1 :
                        temp = torch.narrow(input, 2,  i*chunk_i_y, chunk_i_y +1)
                        temp2 = torch.narrow(temp, 3,  j*chunk_i_x -1 , chunk_i_x +1)
                        print("======================!!! temp size ", temp.size())
                        print("======================!!! temp2 size ", temp2.size())
                       


                    elif i != 0 and i != self.partition_degree_vect[0] -1 and j == 0 :
                        temp = torch.narrow(input, 2,  i*chunk_i_y -1 , chunk_i_y +2)
                        temp2 = torch.narrow(temp, 3,  j*chunk_i_x , chunk_i_x +1)
                        print("======================!!! temp size ", temp.size())
                        print("======================!!! temp2 size ", temp2.size())
                        

                    elif i != 0 and i != self.partition_degree_vect[0] -1 and j ==  self.partition_degree_vect[1] -1:
                        temp = torch.narrow(input, 2,  i*chunk_i_y -1 , chunk_i_y +2)
                        temp2 = torch.narrow(temp, 3,  j*chunk_i_x -1 , chunk_i_x +1)
                        print("======================!!! temp size ", temp.size())
                        print("======================!!! temp2 size ", temp2.size())
                      

                    elif i == self.partition_degree_vect[0] -1 and  j == 0 :
                        temp = torch.narrow(input, 2,  i*chunk_i_y -1 , chunk_i_y +1)
                        temp2 = torch.narrow(temp, 3,  j*chunk_i_x  , chunk_i_x +1)
                        print("======================!!! temp size ", temp.size())
                        print("======================!!! temp2 size ", temp2.size())
                        
                    elif i == self.partition_degree_vect[0] -1  and j ==  self.partition_degree_vect[1] -1 :
                        temp = torch.narrow(input, 2,  i*chunk_i_y -1 , chunk_i_y +1)
                        temp2 = torch.narrow(temp, 3,  j*chunk_i_x -1 , chunk_i_x +1)
                        print("======================!!! temp size ", temp.size())
                        print("======================!!! temp2 size ", temp2.size())
                        

                    elif i == self.partition_degree_vect[0] -1 and j != 0 and j != self.partition_degree_vect[1] -1 :
                        temp = torch.narrow(input, 2,  i*chunk_i_y -1 , chunk_i_y +1)
                        temp2 = torch.narrow(temp, 3,  j*chunk_i_x -1 , chunk_i_x +2)
                        print("======================!!! temp size ", temp.size())
                        print("======================!!! temp2 size ", temp2.size())
                       

                    else:
                        temp = torch.narrow(input, 2,  i*chunk_i_y -1 , chunk_i_y +2)
                        temp2 = torch.narrow(temp, 3,  j*chunk_i_x -1 , chunk_i_x +2)
                        print("======================!!! temp size ", temp.size())
                        print("======================!!! temp2 size ", temp2.size())
                    
                    out_list_row.append(F.conv2d(temp2, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups))


                row = torch.cat(out_list_row, dim=3)
                out_list.append(row)
            
            out = torch.cat(out_list, dim=2)
            print("======================!!! out size ", out.size())
            return out

class Net(nn.Module):
    def __init__(self, in_channels=1 ,  out_channels=1 , kernel_size=3 ):
        super(Net, self).__init__()
        self.conv2D_1 = TiledConv2D(in_channels, out_channels, kernel_size, partition_degree_vect=[4,4,1,1])
    
    def forward(self,x):
        result_1 = self.conv2D_1(x)
        return result_1 


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", default=1, type=int, help="Batch")
    parser.add_argument("--m", default=16, type=int, help="M Dimension")
    parser.add_argument("--k", default=16, type=int, help="K Dimension")
    parser.add_argument("--in_channel", default=1, type=int, help="In_Channel")
    parser.add_argument("--out_channel", default=1, type=int, help="Out_Channel")
    parser.add_argument("--kernelsize", default=3, type=int, help="Kernel_size")
    parser.add_argument("--stride", default=1, type=int, help="Stride")
    parser.add_argument("--padding", default=0, type=int, help="Padding")
    parser.add_argument("--dilation", default=1, type=int, help="Dilation")
    parser.add_argument("--groups", default=1, type=int, help="Groups")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs ")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--lr', type=float, default=0.01, help = "Learning rate")
    parser.add_argument('--gamma', type=float, default=0.7, help = "Learning rate decay ")
    parser.add_argument("--lr_step", type=int, default=1, help="Step LR Scheduler after this many epochs")


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model = Net(in_channels=args.in_channel, out_channels=args.out_channel, kernel_size=args.kernelsize)
    

    model.to(device)
    input = torch.rand(args.b, args.in_channel,  args.m, args.k).cuda()

    output = model(input)

    print("input size", input.size())
    print("output size", output.size())

    # m = nn.Conv2d(args.in_channel, args.out_channel, args.kernelsize).cuda()
    # output_ref = m(input)

    r_weight = Parameter(torch.reshape(torch.arange(0, args.out_channel * args.in_channel
                                                                   * args.kernelsize* args.kernelsize, step=1.0, dtype=torch.float), (args.out_channel, args.in_channel, args.kernelsize, args.kernelsize ))).cuda()

    out_ref = F.conv2d(input, r_weight, None, args.stride, args.padding, args.dilation, args.groups).cuda()

  
    print("output_ref size", out_ref.size())

    assert(torch.allclose(output,out_ref, atol=1e-10))
   
if __name__=="__main__":
    main()
