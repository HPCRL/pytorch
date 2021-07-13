import torch
import torch.nn as nn
from uu.utils import memory 
from uu.utils import correctness_check 
from uu.utils import padding_calc

from torch.nn.parameter import Parameter

class Net_ref(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=128, 
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
        self.conv2d_1 = nn.Conv2d(in_channels=128, 
                                  out_channels=64, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0)
                                  )

    def forward(self, x):
        Th = 2
        Tw = 2
        H = 4
        W = 4
        #TODO: need to detect original padding from unmodified graph
        input_tile = padding_calc.get_input_tile([0,0], H, W, Th, Tw, 1, 1, x)
        print("tile input_tile 1 size", input_tile.size())
        out_1 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([0,1], H, W, Th, Tw, 1, 1, x)
        print("tile input_tile 2 size", input_tile.size())
        out_2 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([1,0], H, W, Th, Tw, 1, 1, x)
        print("tile input_tile 3 size", input_tile.size())
        out_3 = self.conv2d_1(input_tile)

        input_tile = padding_calc.get_input_tile([1,1], H, W, Th, Tw, 1, 1, x)
        print("tile input_tile 4 size", input_tile.size())
        out_4 = self.conv2d_1(input_tile)

        print("out_1 size", out_1.size())
        print("out_2 size", out_2.size())
        print("out_3 size", out_3.size())
        print("out_4 size", out_4.size())

        out_row_1 = torch.cat([out_1, out_2], dim=3)
        out_row_2 = torch.cat([out_3, out_4], dim=3)
        out = torch.cat([out_row_1, out_row_2], dim=2)
        return out






def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    w = model.conv2d_1.weight
    model_ref =  Net_ref(w).to(device)

    input = torch.rand(1,128,4,4, requires_grad = True).cuda() 
    print("input shape", input.size())

    out = model(input)
    out_ref = model_ref(input)

    print("out shape", out.size())
    print("out_ref shape", out_ref.size())
    print("~~ check forward correctness ~~")

    print("out", out)
    print("out_ref", out_ref)
    not_same_num = correctness_check.point_wise_compare_4d(1,64,4,4, out, out_ref)
    


if __name__=="__main__":
    main()
