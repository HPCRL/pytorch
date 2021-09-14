from os import XATTR_REPLACE, XATTR_SIZE_MAX
from torch.autograd.grad_mode import no_grad
import torch
import torch.nn as nn
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy
from torch.nn.parameter import Parameter


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1),
                                #   depth=4,
                                #   num_conv=4
                                  )
        self.conv2d_2 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1),
                                #   depth=3,
                                #   num_conv=4
                                  )
        self.conv2d_3 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1),
                                #   depth=2,
                                #   num_conv=4
                                  )   
        
        self.mxp = maxpool2d.cMaxPool2d((2, 2), (2, 2), mdepth=1, num_maxp=1)

        self.conv2d_4 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1),
                                #   depth=1,
                                #   num_conv=4
                                  )   

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.block1 = sequential.mSequential(*[self.conv2d_1, self.conv2d_2,\
                                                self.mxp, self.conv2d_3, self.conv2d_4])
        
    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).is_cuda
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, 1, 1)
        #print("!!!!!!!", model_device)
        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        for i in range(1,2): 
            for j in range(1,2):
                coord = [i,j]
                print(coord)
                # TODO: here we have to somehow provide static info and num_conv. 
                stream_structure = self.block1
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict)
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                input_tile = self.tsplit(info, x, stream_structure[0], model_device)
                print("input tile", input_tile.size())
                out_temp = self.conv2d_1(input_tile, info)
                print("1 out_temp", out_temp[0].size())
                out_temp = self.conv2d_2(out_temp)
                print("2 out_temp", out_temp[0].size())
                    

                out_temp = self.mxp(out_temp)
                print("max ", out_temp[0].size())
                out_temp = self.conv2d_3(out_temp)
                print("3 out_temp", out_temp[0].size())
                out_temp = self.conv2d_4(out_temp)
                print("4 out_temp", out_temp[0].size())
                
        # # #         # use customized copy
        #         tile_size = [info[1].pt_size[2], info[1].pt_size[3]]
        #         out = self.tcopy(out_temp, out, coord, tile_size)
            
        return out

def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    H = 64 
    W = 64
    nTh = 4
    nTw = 4
    input = torch.rand(1,1,H,W, requires_grad = True)
    
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    out = model(input, H, W, nTh, nTw )
    
    print("done")
    

if __name__=="__main__":
    main()