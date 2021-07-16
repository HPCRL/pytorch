import torch
import torch.nn as nn
from uu.utils import memory 
from uu.utils import correctness_check 
from uu.utils import padding_calc

from torch.nn.parameter import Parameter

class Net_ref(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, 
                                  out_channels=64, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )
        self.conv2d_1.weight = w

    def forward(self, x):
        out = self.conv2d_1(x)
        return out



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, 
                                  out_channels=64, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0)
                                  )

    def forward(self, x):
        Th = 27
        Tw = 27
        H = 81
        W = 81
        #TODO: need to detect original padding from unmodified graph
        input_tile = padding_calc.get_input_tile([0,0], H, W, Th, Tw, 1, 1, x)
        out_1 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([0,1], H, W, Th, Tw, 1, 1, x)
        out_2 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([0,2], H, W, Th, Tw, 1, 1, x)
        out_3 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([1,0], H, W, Th, Tw, 1, 1, x)
        out_4 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([1,1], H, W, Th, Tw, 1, 1, x)
        out_5 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([1,2], H, W, Th, Tw, 1, 1, x)
        out_6 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([2,0], H, W, Th, Tw, 1, 1, x)
        out_7 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([2,1], H, W, Th, Tw, 1, 1, x)
        out_8 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([2,2], H, W, Th, Tw, 1, 1, x)
        out_9 = self.conv2d_1(input_tile)


        out_row_1 = torch.cat([out_1, out_2, out_3], dim=3)
        out_row_2 = torch.cat([out_4, out_5, out_6], dim=3)
        out_row_3 = torch.cat([out_7, out_8, out_9], dim=3)
        out = torch.cat([out_row_1, out_row_2, out_row_3], dim=2)
        return out


def main():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    w = model.conv2d_1.weight
    model_ref =  Net_ref(w).to(device)

    input = torch.rand(1,3,81,81, requires_grad = True).cuda() 
    print("input shape", input.size())

    out = model(input)
    out_ref = model_ref(input)

    print("out shape", out.size())
    print("out_ref shape", out_ref.size())
    print("~~ check forward correctness ~~")

    # print("out", out)
    # print("out_ref", out_ref)
    not_same_num = correctness_check.point_wise_compare_4d(1,64,81,81, out, out_ref)
    
    out.sum().backward()
    out_ref.sum().backward()

    #print("model.conv2d_1.weight.grad", model.conv2d_1.weight.grad)
    #print("model_ref.conv2d_1.weight.grad", model_ref.conv2d_1.weight.grad)
    assert(torch.all(torch.eq(model.conv2d_1.weight.grad, model_ref.conv2d_1.weight.grad)))



if __name__=="__main__":
    main()
