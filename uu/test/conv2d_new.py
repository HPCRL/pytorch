import torch
import torch.nn as nn
from uu.utils import memory 
from uu.utils import correctness_check 
from uu.utils import padding_calc
from uu.layers import conv2d 

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
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0)
                                  )
        self.conv2d_2 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0)
                                  )
        # self.conv2d_3 = conv2d.TiledConv2d(in_channels=64, 
        #                           out_channels=64, 
        #                           kernel_size=(3,3),
        #                           bias = False,
        #                           padding=(1,1)
        #                           )                                                    
        self.block = nn.Sequential(*[self.conv2d_1, self.conv2d_2])

    def forward(self, x):
        return self.block(x)



def main():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    w1 = model.conv2d_1.weight
    w2 = model.conv2d_2.weight

    #print(w1, w2)
    model_ref =  Net_ref(w1, w2).to(device)
    #print(model_ref.conv2d_1.weight, model_ref.conv2d_1.weight)
    input = torch.rand(1,1,3,3, requires_grad = True).cuda()
    print("input shape", input.size())
    print(input)

    out = model(input)
    out_ref = model_ref(input)

    # print("out shape", out.size())
    # print("out_ref shape", out_ref.size())
    # print("~~ check forward correctness ~~")

    # print("out", out)
    # print("out_ref", out_ref)
    # not_same_num = correctness_check.point_wise_compare_4d(1,1,3,3, out, out_ref)
    
    # # out.sum().backward()
    # out_ref.sum().backward()

    # #print("model.conv2d_1.weight.grad", model.conv2d_1.weight.grad)
    # #print("model_ref.conv2d_1.weight.grad", model_ref.conv2d_1.weight.grad)
    # assert(torch.all(torch.eq(model.conv2d_1.weight.grad, model_ref.conv2d_1.weight.grad)))



if __name__=="__main__":
    main()
