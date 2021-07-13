import torch
import torch.nn as nn
from uu.utils import memory 
from uu.utils import correctness_check 


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
                                  padding=(1,1)
                                  )

    def forward(self, x):
        out = self.conv2d_1(x)
        return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    w = model.conv2d_1.weight
    model_ref =  Net_ref(w).to(device)

    input = torch.rand(1,128,4,4, requires_grad = True).cuda() 

    print("input shape", input.size())

    out = model(input)
    out_ref = model(input)

    print("out shape", out.size())
    print("out_ref shape", out_ref.size())
    print("~~ check forward correctness ~~")
    not_same_num = correctness_check.point_wise_compare_4d(1,64,4,4, out, out_ref)
    


if __name__=="__main__":
    main()
