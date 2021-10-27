import torch
import torch.nn as nn
from torch.cuda import init
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 
from uu.utils import memory 
from uu.utils import checkpoint
import time

Kh = 3
Kw = 3
Ph = 1
Pw = 1
chanel = 3
batch = 1
H = 3072
W = 3072
oH = H//32
oW = W//32


class Net_ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, 
                                  out_channels=32, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #0
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))  #1

        self.conv2d_2 = nn.Conv2d(in_channels=32, 
                                  out_channels=64, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #2
        self.maxpool2 = nn.MaxPool2d((2,2), (2,2))  #3
                                
        
        self.conv2d_3 = nn.Conv2d(in_channels=64, 
                                  out_channels=128, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #4
        self.conv2d_4 = nn.Conv2d(in_channels=128, 
                                  out_channels=64, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #5
        self.conv2d_5 = nn.Conv2d(in_channels=64, 
                                  out_channels=128, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #6
        self.maxpool3 = nn.MaxPool2d((2,2), (2,2))  #7

        
        self.conv2d_6 = nn.Conv2d(in_channels=128, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #8
        self.conv2d_7 = nn.Conv2d(in_channels=256, 
                                  out_channels=128, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #9
        self.conv2d_8 = nn.Conv2d(in_channels=128, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)   
                                  )                 #10
        self.maxpool4 = nn.MaxPool2d((2,2), (2,2))  #11

        
        self.conv2d_9 = nn.Conv2d(in_channels=256, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #12
        self.conv2d_10 = nn.Conv2d(in_channels=512, 
                                  out_channels=256, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #13
        self.conv2d_11 = nn.Conv2d(in_channels=256, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #14
        self.conv2d_12 = nn.Conv2d(in_channels=512, 
                                  out_channels=256, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #15
        self.conv2d_13 = nn.Conv2d(in_channels=256, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #16
        self.maxpool5 = nn.MaxPool2d((2,2), (2,2))  #17


        self.conv2d_14 = nn.Conv2d(in_channels=512, 
                                  out_channels=1024, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #18
        self.conv2d_15 = nn.Conv2d(in_channels=1024, 
                                  out_channels=512, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #19
        self.conv2d_16 = nn.Conv2d(in_channels=512, 
                                  out_channels=1024, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #20
        self.conv2d_17 = nn.Conv2d(in_channels=1024, 
                                  out_channels=512, 
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )                 #21
        self.conv2d_18 = nn.Conv2d(in_channels=512, 
                                  out_channels=1024, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )                 #22
        self.conv2d_19 = nn.Conv2d(in_channels=1024, 
                                  out_channels=2048, #28269,        ##fake one number
                                  kernel_size=(1,1),
                                  bias = False,
                                  padding=(0,0)
                                  )    
        self.block1 = nn.Sequential(*[self.conv2d_1, self.maxpool1 ,self.conv2d_2, self.maxpool2, \
                                                self.conv2d_3,  self.conv2d_4, self.conv2d_5, self.maxpool3,  \
                                                self.conv2d_6, self.conv2d_7, self.conv2d_8, self.maxpool4, \
                                                self.conv2d_9, self.conv2d_10, self.conv2d_11, self.conv2d_12, self.conv2d_13, self.maxpool5, \
                                                self.conv2d_14, self.conv2d_15, self.conv2d_16, self.conv2d_17, self.conv2d_18, self.conv2d_19 ]) 

    def forward(self, x):
        out = self.block1(x)
      
        return out




def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float32)
    
    input = torch.rand(batch,chanel,H,W, requires_grad = True)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memUsage = memory.MeasureMemory(device)
    print("==== init ...")
    initmem = memUsage.currentValue()
    print(memory.MemSize(initmem))      
    print(memUsage.available())


    model_ref =  Net_ref().to(device)

    print("==== after load model ...")
    initmem = memUsage.currentValue()
    print(memory.MemSize(initmem))      
    print(memUsage.available())
    input_ref = input.data.clone() 

    start_time = time.time()    
    input_ref = input_ref.cuda()
    input_ref.requires_grad = True
    out_ref = model_ref(input_ref)
    torch.cuda.synchronize()
    ref_fwd_done = time.time()
    ref_elapsed_fwd = ref_fwd_done - start_time

   
    print("==== ref_fwd done ...")
    ref_fwd_use = memUsage.currentValue()-initmem
    print(memory.MemSize(ref_fwd_use) )    
    print("avail ref",memUsage.available())
    print("max ref", memUsage.maxx(), memUsage.maximumValue())


    out_ref.sum().backward()
    print("out_ref", out_ref.size())
    torch.cuda.synchronize()
    ref_elapsed_bwk = time.time()-ref_fwd_done
    ref_elapsed_total = time.time() - start_time
    print("done ref bkw")
    print("\n&& {}, {}, {}\n".format(ref_elapsed_fwd, ref_elapsed_bwk, ref_elapsed_total) )
    
    print("==== ref_bwd done ...")
    ref_bwd_use = memUsage.currentValue()-ref_fwd_use
    ref_bwd_use_total = memUsage.currentValue()-initmem
    print("ref_bwd_use",memory.MemSize(ref_bwd_use))      
    print("ref_bwd_use t", memory.MemSize(ref_bwd_use_total))     
    print("avail ref", memUsage.available())
    print("max ref", memUsage.maxx(),  memUsage.maximumValue())
    print("input graad", input_ref.grad[0,0,0,17])



if __name__=="__main__":
    main()