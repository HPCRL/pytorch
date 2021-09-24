from torch.autograd.grad_mode import no_grad
import torch
import torch.nn as nn
from uu.utils import memory 
from uu.utils import correctness_check 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy
from torch.nn.parameter import Parameter
import time

class Net_ref(nn.Module):
    def __init__(self, w1):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d((2,2), (2,2))
        self.conv1 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0)
                                  )
        self.conv1.weight = Parameter(w1)

    def forward(self, x):
        out = self.maxpool(x)
        out = self.conv1(out)
        #print("ref mxp out\n", out)
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mxp = maxpool2d.cMaxPool2d((2, 2), (2, 2))
        self.conv1 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0)
                                  )
        
    def forward(self, x, H, W, nTh, nTw):
        info = None
        out = self.mxp(x, info)
        out = self.conv1(out)
        return out

def main():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    

    H = 512 
    W = 512
    nTh = 1
    nTw = 1
    input = torch.rand(1,1,H,W)
    input = input.to(device)
    input.requires_grad = True

    our_start = time.time()
    out = model(input, H, W, nTh, nTw )
    out.sum().backward()    
    our_end = time.time()
    #print("our time", our_end-our_start)
    w1 = model.conv1.weight.data
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    
    
    model_ref =  Net_ref(w1).to(device)
    input_ref = input.data
    input_ref = input_ref
    input_ref.requires_grad = True
    input_ref = input_ref.to(device)

    ref_start = time.time() 
    out_ref = model_ref(input_ref)
    out_ref.sum().backward()    
    ref_end = time.time()
    #print("ref time", ref_end-ref_start)
    
    
    # print("input \n", input)
    # print("out shape \n", out)
    # print("out_ref \n", out_ref)
    Hout = out.size()[2]
    Wout = Hout
    print("~~ check forward correctness ~~")
    not_same_num = correctness_check.point_wise_compare_4d(1,1,Hout, Wout, out, out_ref)
    
    
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    # print(input.grad)
    # print(input_ref.grad)
    print("~~ check bkw correctness ~~")
    not_same_num = correctness_check.point_wise_compare_4d(1,1,H, W, input.grad, input_ref.grad)
    
    print("DONE")

if __name__=="__main__":
    main()