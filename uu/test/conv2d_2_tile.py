import torch
import torch.nn as nn
from uu.utils import memory 
from uu.utils import correctness_check 
from uu.utils import padding_calc

from torch.nn.parameter import Parameter

class Net_ref(nn.Module):
    def __init__(self, w1, w2, w3=None):
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
        # self.conv2d_3 = nn.Conv2d(in_channels=64, 
        #                           out_channels=64, 
        #                           kernel_size=(3,3),
        #                           bias = False,
        #                           padding=(1,1)
        #                           )                                                    
        self.conv2d_1.weight = w1
        self.conv2d_2.weight = w2
        #self.conv2d_3.weight = w3

    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.conv2d_2(out)
        #out = self.conv2d_3(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0)
                                  )
        # self.conv2d_3 = nn.Conv2d(in_channels=64, 
        #                           out_channels=64, 
        #                           kernel_size=(3,3),
        #                           bias = False,
        #                           padding=(1,1)
        #                           )                                                    
        self.block = nn.Sequential(*[self.conv2d_1, self.conv2d_2])

    def forward(self, x):
        Th = 1
        Tw = 1
        H = 3
        W = 3
        num_conv = 2
        #TODO: need to detect original padding from unmodified graph
        input_tile = padding_calc.get_input_tile([0,0], H, W, Th, Tw, 1, 1, x, num_conv)
        print("T1 ---", input_tile)
        out_1 = self.block(input_tile)
        print("T1 ---", out_1)

        input_tile = padding_calc.get_input_tile([0,1], H, W, Th, Tw, 1, 1, x, num_conv)
        out_2 = self.block(input_tile)

        input_tile = padding_calc.get_input_tile([0,2], H, W, Th, Tw, 1, 1, x, num_conv)
        out_3 = self.block(input_tile)

        input_tile = padding_calc.get_input_tile([1,0], H, W, Th, Tw, 1, 1, x, num_conv)
        out_4 = self.block(input_tile)

        input_tile = padding_calc.get_input_tile([1,1], H, W, Th, Tw, 1, 1, x, num_conv)
        out_5 = self.block(input_tile)

        input_tile = padding_calc.get_input_tile([1,2], H, W, Th, Tw, 1, 1, x, num_conv)
        out_6 = self.block(input_tile)

        input_tile = padding_calc.get_input_tile([2,0], H, W, Th, Tw, 1, 1, x, num_conv)
        out_7 = self.block(input_tile)

        input_tile = padding_calc.get_input_tile([2,1], H, W, Th, Tw, 1, 1, x, num_conv)
        out_8 = self.block(input_tile)

        input_tile = padding_calc.get_input_tile([2,2], H, W, Th, Tw, 1, 1, x, num_conv)
        out_9 = self.block(input_tile)


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

    #print(w1, w2)
    model_ref =  Net_ref(w1, w2).to(device)
    #print(model_ref.conv2d_1.weight, model_ref.conv2d_1.weight)
    input = torch.rand(1,1,3,3, requires_grad = True)
    print("input shape", input.size())
    print(input)

    out = model(input)
    out_ref = model_ref(input)

    print("out shape", out.size())
    print("out_ref shape", out_ref.size())
    print("~~ check forward correctness ~~")

    print("out", out)
    print("out_ref", out_ref)
    not_same_num = correctness_check.point_wise_compare_4d(1,1,3,3, out, out_ref)
    
    out.sum().backward()
    # out_ref.sum().backward()

    # #print("model.conv2d_1.weight.grad", model.conv2d_1.weight.grad)
    # #print("model_ref.conv2d_1.weight.grad", model_ref.conv2d_1.weight.grad)
    # assert(torch.all(torch.eq(model.conv2d_1.weight.grad, model_ref.conv2d_1.weight.grad)))



if __name__=="__main__":
    main()
