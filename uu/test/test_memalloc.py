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
from torch.utils.checkpoint import checkpoint_sequential



Kh = 3
Kw = 3
Ph = 1
Pw = 1
chanel = 3
batch = 1
H = 2048
W = 2048


class Net_ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, 
                                  out_channels=16, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=16, 
                                  out_channels=16, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))

        self.conv2d_3 = nn.Conv2d(in_channels=16, 
                                  out_channels=32, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_4 = nn.Conv2d(in_channels=32, 
                                  out_channels=32, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.block1 = nn.Sequential(*[self.conv2d_1, self.conv2d_2, self.maxpool1, self.conv2d_3, self.conv2d_4])
       

    def forward(self, x):
        print("ref")
        out = checkpoint_sequential(self.block1, 5, x)
        return out




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=3, 
                                  out_channels=16, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  )  

        self.conv2d_2 = conv2d.TiledConv2d(in_channels=16, 
                                        out_channels=16, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        ) 
        self.mxp1 = maxpool2d.cMaxPool2d((2, 2), (2, 2))
        self.conv2d_3 = conv2d.TiledConv2d(in_channels=16, 
                                  out_channels=32, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  )  

        self.conv2d_4 = conv2d.TiledConv2d(in_channels=32, 
                                        out_channels=32, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        ) 

        

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.block1 = sequential.mSequential(*[self.tsplit, self.conv2d_1, self.conv2d_2, self.mxp1, self.conv2d_3, self.conv2d_4]) #
        
    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, batch, chanel)
        print("!!!!!!!", N, C, oH, oW)
        stream_structure = self.block1
    
        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
    

        info_big_dict = {}
        out_temp_buffer_dict = {}
        #5 shape cor, up, left, right, bot, mid

        #unroll here
        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = (i,j)
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict)
                info_big_dict[coord] = info

                fake_pi = info[0][-11]
                tile_shape = fake_pi.cur_output_shape
                tile_shape = [N,C]+list(tile_shape) #4D output tile

                # if (i == 0 and j == 0) or (i == 0 and j == nTw-1) or (i == nTh-1 and j == 0) or (i == nTh-1 and j == nTw-1):
                #     #corner
                #     if "cor" not in out_temp_buffer_dict:
                #         out_temp_buffer = torch.zeros(*tile_shape).cuda()
                #         out_temp_buffer_dict["cor"] = out_temp_buffer
                # elif (i == 0):
                #     # up
                #     if "up" not in  out_temp_buffer_dict:
                #         out_temp_buffer = torch.zeros(*tile_shape).cuda()
                #         out_temp_buffer_dict["up"] = out_temp_buffer
                # elif (i == nTh-1):
                #     # up
                #     if "bot" not in out_temp_buffer_dict:
                #         out_temp_buffer = torch.zeros(*tile_shape).cuda()
                #         out_temp_buffer_dict["bot"] = out_temp_buffer
                # elif (j == 0):
                #     # up
                #     if "left" not in out_temp_buffer_dict:
                #         out_temp_buffer = torch.zeros(*tile_shape).cuda()
                #         out_temp_buffer_dict["left"] = out_temp_buffer
                # elif (j == nTw-1):
                #     # up
                #     if "right" not in out_temp_buffer_dict:
                #         out_temp_buffer = torch.zeros(*tile_shape).cuda()
                #         out_temp_buffer_dict["right"] = out_temp_buffer
                # else:
                #     if "mid" not in out_temp_buffer_dict:
                #         out_temp_buffer = torch.zeros(*tile_shape).cuda()
                #         out_temp_buffer_dict["mid"] = out_temp_buffer
        
        print("*****", out_temp_buffer_dict.keys())

        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = (i,j)
                # print("++++++++++++++++++++++++++++++++++++++++++++++++")
                # print("coord", coord)
                # TODO: retriev info based on coord and buffer
                info = info_big_dict[coord]
                # if (i == 0 and j == 0) or (i == 0 and j == nTw-1) or (i == nTh-1 and j == 0) or (i == nTh-1 and j == nTw-1):
                #     out_temp = out_temp_buffer_dict["cor"]
                # elif (i == 0):
                #     # up
                #     out_temp = out_temp_buffer_dict["up"]
                # elif (i == nTh-1):
                #     # bot
                #     out_temp = out_temp_buffer_dict["bot"]
                # elif (j == 0):
                #     # left
                #    out_temp = out_temp_buffer_dict["left"]
                # elif (j == nTw-1):
                #     # right
                #     out_temp = out_temp_buffer_dict["right"]
                # else:
                #     out_temp = out_temp_buffer_dict["mid"]




                #chkpoint
                out_temp = checkpoint.checkpoint(self.block1, x, info, stream_structure[1], model_device, [nTh, nTw])
                
                # input_tile = self.tsplit(x, info, stream_structure[1], model_device, [nTh, nTw]) # -1 here is to match 0-base
                # out_temp = self.block1(x, info, stream_structure[1], model_device, [nTh, nTw])

                # use customized copy
                fake_pi = info[0][-11]
                tile_shape = fake_pi.cur_output_shape
                tile_size = [tile_shape[0], tile_shape[1]]

                #print("out_temp size, tile size", out_temp.size(), tile_size, tile_shape)

                output_index = fake_pi.input_slice
                out = self.tcopy(out_temp, out, output_index, tile_size)
                del out_temp
                del info

        return out

def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    input = torch.rand(batch,chanel,H,W, requires_grad = True)   
    model_ref =  Net_ref().to(device)

    if IS_REF==1:
        input_ref = input.data.clone() 
        input_ref = input_ref.cuda()
        input_ref.requires_grad = True
        out_ref = model_ref(input_ref)
        #out_ref.sum().backward()


######################################################
    else:
        out = model(input, H, W, nTh, nTw )
        #time.sleep(5)
        print("######################################################")

        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()
        out.sum().backward()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
        
   

import time
import sys

if __name__=="__main__":
    IS_REF = int(sys.argv[1])
    nTw = int(sys.argv[2])
    nTh=nTw
    main()