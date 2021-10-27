import torch
import torch.nn as nn
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy, relu
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 
from uu.utils import checkpoint
from uu.utils import memory 
from uu.utils import checkpoint
import time

Kh = 3
Kw = 3
Ph = 1
Pw = 1
chanel = 3
batch = 1
H = 1024
W = 1024
oH = H//32
oW = W//32
nTh = 8
nTw = 8

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=3,
                                 out_channels=32,
                                 kernel_size=(Kh,Kw),
                                 bias = False,
                                 padding=(Ph,Pw)
                                 )                 #0
        self.maxpool1 = maxpool2d.cMaxPool2d((2,2), (2,2))  #1
    
        self.conv2d_2 = conv2d.TiledConv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #2
        self.maxpool2 = maxpool2d.cMaxPool2d((2,2), (2,2))  #3
                                
        
        self.conv2d_3 = conv2d.TiledConv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #4
        self.conv2d_4 = conv2d.TiledConv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #5
        self.conv2d_5 = conv2d.TiledConv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #6
        self.maxpool3 = maxpool2d.cMaxPool2d((2,2), (2,2))  #7
    
        
        self.conv2d_6 = conv2d.TiledConv2d(in_channels=128,
                                    out_channels=256,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #8
        self.conv2d_7 = conv2d.TiledConv2d(in_channels=256,
                                    out_channels=128,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #9
        self.conv2d_8 = conv2d.TiledConv2d(in_channels=128,
                                    out_channels=256,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)  
                                    )                 #10
        self.maxpool4 = maxpool2d.cMaxPool2d((2,2), (2,2))  #11
    
        
        self.conv2d_9 = conv2d.TiledConv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #12
        self.conv2d_10 = conv2d.TiledConv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #13
        self.conv2d_11 = conv2d.TiledConv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #14
        self.conv2d_12 = conv2d.TiledConv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #15
        self.conv2d_13 = conv2d.TiledConv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #16
        self.maxpool5 = maxpool2d.cMaxPool2d((2,2), (2,2))  #17
    
    
        self.conv2d_14 = conv2d.TiledConv2d(in_channels=512,
                                    out_channels=1024,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #18
        self.conv2d_15 = conv2d.TiledConv2d(in_channels=1024,
                                    out_channels=512,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #19
        self.conv2d_16 = conv2d.TiledConv2d(in_channels=512,
                                    out_channels=1024,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #20
        self.conv2d_17 = conv2d.TiledConv2d(in_channels=1024,
                                    out_channels=512,
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #21
        self.conv2d_18 = conv2d.TiledConv2d(in_channels=512,
                                    out_channels=1024,
                                    kernel_size=(Kh,Kw),
                                    bias = False,
                                    padding=(Ph,Pw)
                                    )                 #22
        self.conv2d_19 = conv2d.TiledConv2d(in_channels=1024,
                                    out_channels=28269,        ##fake one number
                                    kernel_size=(1,1),
                                    bias = False,
                                    padding=(0,0)
                                    )                 #23
        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.block1 = sequential.mSequential(*[self.tsplit, self.conv2d_1, self.maxpool1 ,self.conv2d_2, self.maxpool2, \
                                                self.conv2d_3,  self.conv2d_4, self.conv2d_5, self.maxpool3,  \
                                                self.conv2d_6, self.conv2d_7, self.conv2d_8, self.maxpool4, \
                                                self.conv2d_9, self.conv2d_10, self.conv2d_11, self.conv2d_12, self.conv2d_13, self.maxpool5, \
                                                self.conv2d_14, self.conv2d_15, self.conv2d_16, self.conv2d_17, self.conv2d_18, self.conv2d_19]) #
        
    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, batch, chanel)
        #print("!!!!!!!", model_device)
        print("!!!!!!!", N, C, oH, oW)
        stream_structure = self.block1

        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = [i,j]
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict)
                
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                print("coord", coord)
                #out_temp = self.block1(x, info, stream_structure[1], model_device, [nTh, nTw])
                out_temp = checkpoint.checkpoint(self.block1, x, info, stream_structure[1], model_device, [nTh, nTw])
               
                # use customized copy
                fake_pi = info[0][-11]
                tile_shape = fake_pi.cur_output_shape
                tile_size = [tile_shape[0], tile_shape[1]]
                output_index = fake_pi.input_slice

                # print(tile_shape, tile_size, output_index)
                out = self.tcopy(out_temp, out, output_index, tile_size)
                del out_temp
                del info
        
        #out = nn.Softmax(out)
        return out

def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    memUsage = memory.MeasureMemory(device)
    print("==== our init ...")
    our_initmem = memUsage.currentValue()
    print(memory.MemSize(our_initmem))     
    print(memUsage.available())


    input = torch.rand(batch,chanel,H,W, requires_grad = True)
    out = model(input, H, W, nTh, nTw )

    print("==== our_fwd done ...")
    our_fwd_use = memUsage.currentValue()-our_initmem
    print(memory.MemSize(our_fwd_use) )     
    print("avail our",memUsage.available())
    print("max our", memUsage.maxx(), memUsage.maximumValue())

    out.sum().backward()
    print("out", out.size())
    print("==== our_bwd done ...")
    our_bwd_use = memUsage.currentValue()-our_fwd_use
    our_bwd_use_total = memUsage.currentValue()-our_initmem
    print("our_bwd_use", memory.MemSize(our_bwd_use))   
    print("our_bwd_use t", memory.MemSize(our_bwd_use_total))    
    print("avail our",memUsage.available())
    print("max our", memUsage.maxx(), memUsage.maximumValue())


   
if __name__=="__main__":
    main()