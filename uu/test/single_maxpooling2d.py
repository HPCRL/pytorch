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
    def __init__(self):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d((2,2), (2,2))

    def forward(self, x):
        out = self.maxpool(x)
        #print("ref mxp out\n", out)
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mxp = maxpool2d.cMaxPool2d((2, 2), (2, 2))

    def forward(self, x, H, W, nTh, nTw):
        info = None
        out = self.mxp(x, info)
        return out

def main():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    
    model_ref =  Net_ref().to(device)

    H = 4096 
    W = 4096
    nTh = 1
    nTw = 1
    input = torch.rand(1,1,H,W, requires_grad = True)

    input_ref = input.data
    input_ref = input_ref.cuda()
    input_ref.requires_grad = True

    ref_start = time.time() 
    out_ref = model_ref(input_ref)
    ref_end = time.time()
    print("ref time", ref_end-ref_start)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    our_start = time.time()
    out = model(input, H, W, nTh, nTw )[0]
    our_end = time.time()
    print("our time", our_end-our_start)

    print("out shape", out)
    print("out_ref ", out_ref)
    print("~~ check forward correctness ~~")

    not_same_num = correctness_check.point_wise_compare_4d(1,1,H//2, W//2, out, out_ref)
    # out_ref.sum().backward()
    # print(input_ref.grad)
    
    # out.sum().backward()
    # print(input.grad)

if __name__=="__main__":
    main()