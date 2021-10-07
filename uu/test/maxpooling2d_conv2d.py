from torch.autograd.grad_mode import no_grad
import torch
import torch.nn as nn
from uu.utils import memory 
from uu.utils import correctness_check 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy
from torch.nn.parameter import Parameter

class Net_ref(nn.Module):
    def __init__(self, w1, w2, w3, w4, w5):
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
                                
        self.maxpool = nn.MaxPool2d((2,2), (2,2))

        self.conv2d_3 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )                            

        self.conv2d_4 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )
        self.conv2d_5 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1)
                                  )                          

        self.conv2d_1.weight = Parameter(w1)
        self.conv2d_2.weight = Parameter(w2)
        self.conv2d_3.weight = Parameter(w3)
        self.conv2d_4.weight = Parameter(w4)
        self.conv2d_5.weight = Parameter(w5)

    def forward(self, x):
        out = self.conv2d_1(x)
        #print("ref 1st out\n", out)
        out = self.conv2d_2(out)
        #print("ref 2nd out\n", out)
        

        out = self.maxpool(out)
        #print("ref mxp out\n", out)

        out = self.conv2d_3(out)
        #print("ref 3rd out\n", out)

        out = self.conv2d_4(out)
        #print("ref 4th out\n", out)
        out = self.conv2d_5(out)
        # print("ref 5th out\n", out)

        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0),
                                  depth=4,
                                  num_conv=5
                                  )
        self.conv2d_2 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0),
                                  depth=3,
                                  num_conv=5
                                  )
        
        self.mxp = maxpool2d.cMaxPool2d((2, 2), (2, 2), is_last=False)

        self.conv2d_3 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0),
                                  depth=2,
                                  num_conv=5
                                  )   
        self.conv2d_4 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(0,0),
                                  depth=1,
                                  num_conv=5
                                  )   
        self.conv2d_5 = conv2d.TiledConv2d(in_channels=1, 
                                    out_channels=1, 
                                    kernel_size=(3,3),
                                    bias = False,
                                    padding=(0,0),
                                    depth=0,
                                    num_conv=5
                                    )   
        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()

        # self.block = sequential.mSequential(*[self.conv2d_1, self.conv2d_2, self.conv2d_3, \
        #                                         self.mxp, self.conv2d_4, self.conv2d_5])
        
    def forward(self, x, H, W, nTh, nTw):
        model_device = next(self.parameters()).is_cuda
        # TODO: here we have to somehow infer the shape of the output of the segment. 
        out = torch.zeros(1, 1, H//2, W//2, requires_grad=True).cuda()
       
        for i in range(0,1): 
            for j in range(0,1):
                coord = [i,j]
                print(coord)
                # TODO: here we have to somehow provide static info and num_conv. 
                stream_structure = [("conv2d", 2), ("pooling", 1), ("conv2d", 3)]
                num_conv = 5
                #stream_structure = [("conv2d", 3)]
                info = padding_calc.compute_info_beta([i,j], H, W, nTh, nTw, 1, 1, stream_structure, num_conv)
                #print(info)
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                input_tile = self.tsplit(info, x, num_conv-1, model_device)
                # print("input tile", input_tile.size())
                out_temp = self.conv2d_1(input_tile, info)
                #print("1 out_temp", out_temp[0].size(), out_temp[0])
                out_temp = self.conv2d_2(out_temp)
                #print("2 out_temp", out_temp[0].size(), out_temp[0])
                    

                out_temp = self.mxp(out_temp)
                #print("max ", out_temp[0].size(), out_temp[0])
                out_temp = self.conv2d_3(out_temp)
                #print("3 out_temp", out_temp[0].size(), out_temp[0])

                out_temp = self.conv2d_4(out_temp)
                # print("4 out_temp", out_temp[0].size(), out_temp[0])
                
                #print("5 out_temp", out_temp[0].size(), out_temp[0])
                out_temp = self.conv2d_5(out_temp)
 


        #         print("!!!!!!!!!!!!!!out_temp", out_temp.size(), out_temp)
        #         # default copy has issue in bkward
        #         # out[:,:, i*Th:(i+1)*Th, j*Tw:(j+1)*Tw] = out_temp
                
        #         # use customized copy
                tile_size = [info[0].pt_size[2], info[0].pt_size[3]]
                out = self.tcopy(out_temp, out, coord, tile_size)
                

        return out

def main():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    w1 = model.conv2d_1.weight.data
    w2 = model.conv2d_2.weight.data
    w3 = model.conv2d_3.weight.data
    w4 = model.conv2d_4.weight.data
    w5 = model.conv2d_5.weight.data
    
    model_ref =  Net_ref(w1, w2, w3, w4, w5).to(device)

    H = 224 
    W = 224
    nTh = 1
    nTw = 1
    input = torch.rand(1,1,H,W, requires_grad = True)

    input_ref = input.data
    input_ref = input_ref.cuda()
    out_ref = model_ref(input_ref)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    out = model(input, H, W, nTh, nTw )

    print("out shape", out)
    print("out_ref ", out_ref)
    print("~~ check forward correctness ~~")

    not_same_num = correctness_check.point_wise_compare_4d(1,1,H//2, W//2, out, out_ref)

    

if __name__=="__main__":
    main()