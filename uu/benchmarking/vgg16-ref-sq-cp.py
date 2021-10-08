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
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


Kh = 3
Kw = 3
Ph = 1
Pw = 1
chanel = 3
batch = 1
H = 10240
W = 10240
oH = H//32
oW = W//32

class Net_ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))
        self.conv2d_3 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_4 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.maxpool2 = nn.MaxPool2d((2,2), (2,2))                          
        self.conv2d_5 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_6 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_7 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool3 = nn.MaxPool2d((2,2), (2,2))

        self.conv2d_8 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_9 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_10 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )

        self.maxpool4 = nn.MaxPool2d((2,2), (2,2))

        self.conv2d_11 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_12 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_13 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )

        self.maxpool5 = nn.MaxPool2d((2,2), (2,2))
        
        self.flat = nn.Flatten()
        in_feature = chanel*oH*oW
        self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 1024, bias=False)

        self.block1 = nn.Sequential(*[self.conv2d_1, self.conv2d_2, self.maxpool1, \
                                                self.conv2d_3,  self.conv2d_4, self.maxpool2,  \
                                                self.conv2d_5, self.conv2d_6, self.conv2d_7, self.maxpool3, \
                                                self.conv2d_8, self.conv2d_9, self.conv2d_10, self.maxpool4, \
                                                self.conv2d_11, self.conv2d_12, self.conv2d_13, self.maxpool5, \
                                                self.flat, self.fc1, self.fc2 ]) 


        

    def forward(self, x):
        #out = checkpoint_sequential(self.block1, 4,x)
        out = checkpoint_sequential(self.block1, 5,x)
        return out




def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float64)
    
    input = torch.rand(batch,chanel,H,W, requires_grad = True)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memUsage = memory.MeasureMemory(device)
    print("==== init ...")
    initmem = memUsage.currentValue()
    print(memory.MemSize(initmem))      
    print(memUsage.available())


    model_ref =  Net_ref().to(device)
    input_ref = input.data.clone() 
    input_ref = input_ref.cuda()
    input_ref.requires_grad = True
    out_ref = model_ref(input_ref)

   
    print("==== ref_fwd done ...")
    ref_fwd_use = memUsage.currentValue()-initmem
    print(memory.MemSize(ref_fwd_use) )    
    print("avail ref sq",memUsage.available())
    print("max ref sq", memUsage.maxx(), memUsage.maximumValue())



    print("done ref sq")
    out_ref.sum().backward()
    print("done ref bkw sq")

    print("==== ref_bwd done ...")
    ref_bwd_use = memUsage.currentValue()-ref_fwd_use
    ref_bwd_use_total = memUsage.currentValue()-initmem
    print("ref_bwd_use sq",memory.MemSize(ref_bwd_use))      
    print("ref_bwd_use t sq ", memory.MemSize(ref_bwd_use_total))     
    print("avail ref sq", memUsage.available())
    print("max ref sq", memUsage.maxx(),  memUsage.maximumValue())




if __name__=="__main__":
    main()