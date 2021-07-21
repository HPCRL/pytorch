import torch
import torch.nn as nn
from uu.utils import memory 
from uu.utils import correctness_check 
from uu.utils import padding_calc
from uu.layers import conv2d 

class Net_ref(nn.Module):
    def __init__(self, w1, w2, w3):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )
        self.conv2d_3 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )                                                    
        self.conv2d_1.weight = w1
        self.conv2d_2.weight = w2
        self.conv2d_3.weight = w3

    def forward(self, x):
        out = self.conv2d_1(x)
        #print("ref 1st out\n", out)
        out = self.conv2d_2(out)
        #print("ref 2nd out\n", out)
        out = self.conv2d_3(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: when we rewirte the network, we should know the depth info.
        # depth is 0 if it is the last conv2d, reversely increased
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0),
                                  depth=2
                                  )
        self.conv2d_2 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0),
                                  depth=1
                                  )
        self.conv2d_3 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0),
                                  depth=0
                                  )   
        
        # TODO: How to make sequential work??                                                 
        # self.block = nn.Sequential(*[self.conv2d_1, self.conv2d_2])

    def forward(self, x, H, W, Th, Tw):
        num_conv = 3
        # preprocess network, not sure if need to put it here 
        info = padding_calc.compute_info([0,0], H, W, Th, Tw, 1, 1, x, num_conv)
        # assume we prepare the very first input
        input_tile = padding_calc.get_input_tile(info, x, num_conv-1)
       # print(input_tile.size())
       # print(input_tile)
        out_1 = self.conv2d_1(input_tile, info)
        out_1 = self.conv2d_2(out_1, info)
        out_1 = self.conv2d_3(out_1, info)
        #print("*******", out_1)


        info = padding_calc.compute_info([0,1], H, W, Th, Tw, 1, 1, x, num_conv)
        # assume we prepare the very first input
        input_tile = padding_calc.get_input_tile(info, x, num_conv-1)
       # print(input_tile.size())
       # print(input_tile)
        out_2 = self.conv2d_1(input_tile, info)
        out_2 = self.conv2d_2(out_2, info)
        out_2 = self.conv2d_3(out_2, info)
        #print("*******", out_2)

        info = padding_calc.compute_info([0,2], H, W, Th, Tw, 1, 1, x, num_conv)
        # assume we prepare the very first input
        input_tile = padding_calc.get_input_tile(info, x, num_conv-1)
       # print(input_tile.size())
       # print(input_tile)
        out_3 = self.conv2d_1(input_tile, info)
        out_3 = self.conv2d_2(out_3, info)
        out_3 = self.conv2d_3(out_3, info)
        #print("*******", out_3)


        info = padding_calc.compute_info([1,0], H, W, Th, Tw, 1, 1, x, num_conv)
        # assume we prepare the very first input
        input_tile = padding_calc.get_input_tile(info, x, num_conv-1)
       # print(input_tile.size())
       # print(input_tile)
        out_4 = self.conv2d_1(input_tile, info)
        out_4 = self.conv2d_2(out_4, info)
        out_4 = self.conv2d_3(out_4, info)
        #print("*******", out_4)

        info = padding_calc.compute_info([1,1], H, W, Th, Tw, 1, 1, x, num_conv)
        # assume we prepare the very first input
        input_tile = padding_calc.get_input_tile(info, x, num_conv-1)
       # print(input_tile.size())
       # print(input_tile)
        out_5 = self.conv2d_1(input_tile, info)
        out_5 = self.conv2d_2(out_5, info)
        out_5 = self.conv2d_3(out_5, info)
        #print("*******", out_5)
        
        info = padding_calc.compute_info([1,2], H, W, Th, Tw, 1, 1, x, num_conv)
        # assume we prepare the very first input
        input_tile = padding_calc.get_input_tile(info, x, num_conv-1)
       # print(input_tile.size())
       # print(input_tile)
        out_6 = self.conv2d_1(input_tile, info)
        out_6 = self.conv2d_2(out_6, info)
        out_6 = self.conv2d_3(out_6, info)
        #print("*******", out_6)

        info = padding_calc.compute_info([2,0], H, W, Th, Tw, 1, 1, x, num_conv)
        # assume we prepare the very first input
        input_tile = padding_calc.get_input_tile(info, x, num_conv-1)
       # print(input_tile.size())
       # print(input_tile)
        out_7 = self.conv2d_1(input_tile, info)
        out_7 = self.conv2d_2(out_7, info)
        out_7 = self.conv2d_3(out_7, info)
        #print("*******", out_7)

        info = padding_calc.compute_info([2,1], H, W, Th, Tw, 1, 1, x, num_conv)
        # assume we prepare the very first input
        input_tile = padding_calc.get_input_tile(info, x, num_conv-1)
       # print(input_tile.size())
       # print(input_tile)
        out_8 = self.conv2d_1(input_tile, info)
        out_8 = self.conv2d_2(out_8, info)
        out_8 = self.conv2d_3(out_8, info)
        #print("*******", out_8)

        info = padding_calc.compute_info([2,2], H, W, Th, Tw, 1, 1, x, num_conv)
        # assume we prepare the very first input
        input_tile = padding_calc.get_input_tile(info, x, num_conv-1)
       # print(input_tile.size())
       # print(input_tile)
        out_9 = self.conv2d_1(input_tile, info)
        out_9 = self.conv2d_2(out_9, info)
        out_9 = self.conv2d_3(out_9, info)
        #print("*******", out_9)

        out_row_1 = torch.cat([out_1, out_2, out_3], dim=3)
        out_row_2 = torch.cat([out_4, out_5, out_6], dim=3)
        out_row_3 = torch.cat([out_7, out_8, out_9], dim=3)

        # print("---", out_row_1.size())
        # print("---", out_row_2.size())
        # print("---", out_row_3.size())
        out = torch.cat([out_row_1, out_row_2, out_row_3], dim=2)

        return out


def main():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    w1 = model.conv2d_1.weight
    w2 = model.conv2d_2.weight
    w3 = model.conv2d_3.weight

    #print(w1, w2)
    model_ref =  Net_ref(w1, w2, w3).to(device)
    #print(model_ref.conv2d_1.weight, model_ref.conv2d_2.weight)
    
    H = 3
    W = 3
    Th = int(H/3)
    Tw = int(W/3)
    input = torch.rand(1,1,H,W, requires_grad = True).cuda()
    print("input shape", input.size())
    print(input)



    out_ref = model_ref(input)
    out = model(input, H, W, Th, Tw )
    

    print("out shape", out.size())
    print("out_ref shape", out_ref.size())
    print("~~ check forward correctness ~~")

    print("out", out)
    print("out_ref", out_ref)
    not_same_num = correctness_check.point_wise_compare_4d(1,1,H, W, out, out_ref)
    
    out.sum().backward()
    #out_ref.sum().backward()

    # #print("model.conv2d_1.weight.grad", model.conv2d_1.weight.grad)
    # #print("model_ref.conv2d_1.weight.grad", model_ref.conv2d_1.weight.grad)
    # assert(torch.all(torch.eq(model.conv2d_1.weight.grad, model_ref.conv2d_1.weight.grad)))



if __name__=="__main__":
    main()
