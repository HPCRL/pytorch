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
nTh = 16
nTw = 16

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  )  

        self.conv2d_2 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        ) 
        self.mxp1 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_3 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_4 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )

        self.mxp2 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_5 = conv2d.TiledConv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  )  

        self.conv2d_6 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        ) 
        self.conv2d_7 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.mxp3 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_8 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_9 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_10 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )

        self.mxp4 = maxpool2d.cMaxPool2d((2, 2), (2, 2))



        self.conv2d_11 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_12 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_13 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )

        self.mxp5 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        in_feature = chanel*oH*oW
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 1024, bias=False)

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.block1 = sequential.mSequential(*[self.tsplit, self.conv2d_1, self.conv2d_2, self.mxp1, \
                                                self.conv2d_3,  self.conv2d_4, self.mxp2,  \
                                                self.conv2d_5, self.conv2d_6, self.conv2d_7, self.mxp3, \
                                                self.conv2d_8, self.conv2d_9, self.conv2d_10, self.mxp4, \
                                                self.conv2d_11, self.conv2d_12, self.conv2d_13, self.mxp5,  
                                                ]) #
        
    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, batch, chanel)
        #print("!!!!!!!", model_device)
        stream_structure = self.block1

        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        print("out shape", out.size())
        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = [i,j]
                print("coord", coord)
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict)
    
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                out_temp = checkpoint.checkpoint(self.block1, x, info, stream_structure[1], model_device, [nTh, nTw])


                # use customized copy
                fake_pi = info[0][-11]
                tile_shape = fake_pi.cur_output_shape
                tile_size = [tile_shape[0], tile_shape[1]]
                output_index = fake_pi.input_slice
                #print(tile_shape, tile_size, output_index)
                out = self.tcopy(out_temp, out, output_index, tile_size)
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # memUsage = memory.MeasureMemory(device)
                # print("==== loop ...", coord)
                # initmem = memUsage.currentValue()
                # print(memory.MemSize(initmem))      #now should be around 3.8MB
                del info

        out = self.flat(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)


    input = torch.rand(batch,chanel,H,W, requires_grad = True)
    memUsage = memory.MeasureMemory(device)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")

    print("==== our init ...")
    our_initmem = memUsage.currentValue()
    print(memory.MemSize(our_initmem))     
    print(memUsage.available())
    out = model(input, H, W, nTh, nTw )

    print("==== our_fwd done ...")
    our_fwd_use = memUsage.currentValue()-our_initmem
    print(memory.MemSize(our_fwd_use) )     
    print("avail our",memUsage.available())
    print("max our", memUsage.maxx(), memUsage.maximumValue())

   
    #print(input_ref.grad)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    out.sum().backward()
    print("==== our_bwd done ...")
    our_bwd_use = memUsage.currentValue()-our_fwd_use
    our_bwd_use_total = memUsage.currentValue()-our_initmem
    print("our_bwd_use", memory.MemSize(our_bwd_use))   
    print("our_bwd_use t", memory.MemSize(our_bwd_use_total))    
    print("avail our",memUsage.available())
    print("max our", memUsage.maxx(), memUsage.maximumValue())


if __name__=="__main__":
    main()