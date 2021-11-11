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



class Net_ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, 
                                  out_channels=64, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=64, 
                                  out_channels=64, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))
        self.conv2d_3 = nn.Conv2d(in_channels=64, 
                                  out_channels=128, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_4 = nn.Conv2d(in_channels=128, 
                                  out_channels=128, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.maxpool2 = nn.MaxPool2d((2,2), (2,2))                          
        self.conv2d_5 = nn.Conv2d(in_channels=128, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_6 = nn.Conv2d(in_channels=256, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_7 = nn.Conv2d(in_channels=256, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool3 = nn.MaxPool2d((2,2), (2,2))

        self.conv2d_8 = nn.Conv2d(in_channels=256, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_9 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_10 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )

        self.maxpool4 = nn.MaxPool2d((2,2), (2,2))

        self.conv2d_11 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_12 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_13 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )

        self.maxpool5 = nn.MaxPool2d((2,2), (2,2))
        self.relu = nn.ReLU()
        # self.flat = nn.Flatten()
        # in_feature = 512*oH*oW
        # self.fc1 = nn.Linear(in_feature, 4096, bias=False)
        # self.fc2 = nn.Linear(4096, 4096, bias=False)
        # self.fc3 = nn.Linear(4096, 1000, bias=False)

        self.avgp = nn.AvgPool2d(oH, stride=1)
        self.flat = nn.Flatten()
        in_feature = 512
        self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        self.sft = nn.Softmax(dim=-1)



        self.block1 = nn.Sequential(*[self.conv2d_1, self.relu, self.conv2d_2, self.relu,self.maxpool1, \
                                                self.conv2d_3, self.relu, self.conv2d_4, self.relu, self.maxpool2,  \
                                                self.conv2d_5, self.relu, self.conv2d_6, self.relu, self.conv2d_7, self.relu, self.maxpool3, \
                                                self.conv2d_8, self.relu, self.conv2d_9, self.relu, self.conv2d_10, self.relu, self.maxpool4, \
                                                self.conv2d_11, self.relu, self.conv2d_12, self.relu, self.conv2d_13, self.relu, self.maxpool5, \
                                                self.avgp, self.flat, self.fc1, self.sft ]) 

    def forward(self, x):
        out = self.block1(x)
        return out




def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float32)

    # add loss function here
    criterion =  nn.MSELoss()
   
    

    #print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # memUsage = memory.MeasureMemory(device)
    # print("==== init ...")
    # initmem = memUsage.currentValue()
    # print(memory.MemSize(initmem))      
    # print(memUsage.available())


    model_ref =  Net_ref().to(device)

    # print("==== after load model ...")
    # initmem = memUsage.currentValue()
    # print(memory.MemSize(initmem))      
    # print(memUsage.available())

    ref_elapsed_fwd = 0
    ref_elapsed_bwk = 0
    start_time = time.time()    
    for i in range(0,1):
        input_ref = torch.rand(batch,chanel,H,W)
        labels = torch.rand(batch, 1024).cuda()
        
        local_start = time.time() 
        input_ref = input_ref.cuda()
        input_ref.requires_grad = True
        out_ref = model_ref(input_ref)
        # torch.cuda.synchronize()
        # ref_fwd_done = time.time()
        # ref_elapsed_fwd += (ref_fwd_done - local_start)


    #print("==== ref_fwd done ...")
    # ref_fwd_use = memUsage.currentValue()-initmem
    # print(memory.MemSize(ref_fwd_use) )    
    # print("avail ref",memUsage.available())
    # print("max ref", memUsage.maxx(), memUsage.maximumValue())

        loss = criterion(out_ref, labels)
        loss.backward()
        # torch.cuda.synchronize()
        # ref_elapsed_bwk += (time.time()-ref_fwd_done)


    
    torch.cuda.synchronize()    
    ref_elapsed_total = time.time() - start_time
    #print("done ref bkw")
    print("\n&& {}\n".format(ref_elapsed_total) )
    
    # print("==== ref_bwd done ...")
    # ref_bwd_use = memUsage.currentValue()-ref_fwd_use
    # ref_bwd_use_total = memUsage.currentValue()-initmem
    # print("ref_bwd_use",memory.MemSize(ref_bwd_use))      
    # print("ref_bwd_use t", memory.MemSize(ref_bwd_use_total))     
    # print("avail ref", memUsage.available())
    # print("max ref", memUsage.maxx(),  memUsage.maximumValue())
    # print("input graad", input_ref.grad[0,0,0,17])

import sys

if __name__=="__main__":
    Kh = 3
    Kw = 3
    Ph = 1
    Pw = 1
    chanel = 3
    batch = 8


    H = int(sys.argv[1])
    W = H
    oH = H//32
    oW = W//32

    main()
