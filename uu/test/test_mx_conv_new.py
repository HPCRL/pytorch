import torch
import torch.nn as nn
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 


def print_grad(self, grad_input, grad_output):
    print('Inside '+ self.__class__.__name__+ ' backward')
    # print('grad_input : ', len(grad_input))
    # print('grad_output : ', len(grad_output))
    print('grad_output size : ', grad_output[0].size())
    print('ref grad_output  :\n ', grad_output[0])
    print('grad_input size : ', grad_input[0].size())
    print('ref grad_input  : \n', grad_input[0])


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
                                  padding=(1,1),
                                  )   
                        

        self.conv2d_1.weight = Parameter(w1)
        self.conv2d_2.weight = Parameter(w2)
        self.conv2d_3.weight = Parameter(w3)
        self.conv2d_4.weight = Parameter(w4)
        self.conv2d_5.weight = Parameter(w5)


        # self.conv2d_1.register_full_backward_hook(print_grad)
        # self.conv2d_2.register_full_backward_hook(print_grad)
        # self.conv2d_3.register_full_backward_hook(print_grad)
        # self.conv2d_4.register_full_backward_hook(print_grad)
        # self.maxpool.register_full_backward_hook(print_grad)

    def forward(self, x):
        out = self.conv2d_1(x)
        #print("ref 1st out\n", out)
        out = self.conv2d_2(out)
        #print("ref 2nd out\n", out)
        out = self.conv2d_3(out)
        #print("ref 3rd out\n", out)
        out = self.maxpool(out)
        #print("ref mxp out\n", out)
        out = self.conv2d_4(out)
        #print("ref 4th out\n", out)
        out = self.conv2d_5(out)
        return out




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1),
                                  )
        self.conv2d_2 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1),
                                  )
        self.conv2d_3 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1),
                                  )   
        
        self.mxp = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_4 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1),
                                  )   
        self.conv2d_5 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(3,3),
                                  bias = False,
                                  padding=(1,1),
                                  )   

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.block1 = sequential.mSequential(*[self.conv2d_1, self.conv2d_2, self.conv2d_3,\
                                                self.mxp,  self.conv2d_4, self.conv2d_5])
        
    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).is_cuda
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, 1, 1)
        #print("!!!!!!!", model_device)
        stream_structure = self.block1

        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        for i in range(0,1): 
            for j in range(0,1):
                coord = [i,j]
                print("coord", coord)
                # TODO: here we have to somehow provide static info and num_conv. 
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict)
                print("++++++++++++++++++++++++++++++++++++++++++++++++", info)
                input_tile = self.tsplit(x, info, stream_structure[0], model_device, [nTh-1, nTw-1]) # -1 here is to match 0-base
                print("***input tile", input_tile.size())
                out_temp = self.conv2d_1(input_tile, info)
                print("1 out_temp", out_temp[0].size())
                out_temp = self.conv2d_2(out_temp)
                print("2 out_temp", out_temp[0].size())
                out_temp = self.conv2d_3(out_temp)
                print("3 out_temp", out_temp[0].size())

                out_temp = self.mxp(out_temp)
                print("max ", out_temp[0].size())
                
                out_temp = self.conv2d_4(out_temp)
                print("4 out_temp", out_temp[0].size())

                out_temp = self.conv2d_5(out_temp)
                print("5 out_temp", out_temp.size())
                
                # use customized copy
                fake_pi = info[0][-11]
                tile_shape = fake_pi.cur_output_shape
                tile_size = [tile_shape[0], tile_shape[1]]
                output_index = fake_pi.input_slice
                print(tile_shape, tile_size, output_index)
                out = self.tcopy(out_temp, out, output_index, tile_size)
                #del info
        return out

def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    H = 16
    W = 16
    nTh = 4
    nTw = 4
    input = torch.rand(1,1,H,W, requires_grad = True)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    out = model(input, H, W, nTh, nTw )

    



    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    w1 = model.conv2d_1.weight.data
    w2 = model.conv2d_2.weight.data
    w3 = model.conv2d_3.weight.data
    w4 = model.conv2d_4.weight.data
    w5 = model.conv2d_5.weight.data
    model_ref =  Net_ref(w1, w2, w3, w4, w5).to(device)
    input_ref = input.data
    input_ref = input_ref.cuda()
    input_ref.requires_grad = True
    out_ref = model_ref(input_ref)
    print("done")

    # print("out shape", out)
    # print("out_ref ", out_ref)
    # print("~~ check forward correctness ~~")
    # not_same_num = correctness_check.point_wise_compare_4d(1,1,H//2, W//2, out, out_ref)
    

if __name__=="__main__":
    main()