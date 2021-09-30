import torch
import torch.nn as nn
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 

li = []
grad_dict_bk = {}
def print_grad(self, grad_input, grad_output):
    print('Inside '+ self.__class__.__name__+ ' backward')
    # print('grad_input : ', len(grad_input))
    # print('grad_output : ', len(grad_output))
    print('grad_output size : ', grad_output[0].size())
    #print('ref grad_output  :\n ', grad_output[0])
    print('grad_input size : ', grad_input[0].size())
    #print('ref grad_input  : \n', grad_input[0])
    li.append( [grad_input, grad_output])

    

Kh = 3
Kw = 3
Ph = 1
Pw = 1

class Net_ref(nn.Module):
    def __init__(self, w1, w2):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))
        self.conv2d_2 = nn.Conv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )

        self.maxpool2 = nn.MaxPool2d((2,2), (2,2))
        self.conv2d_1.weight = Parameter(w1)
        self.conv2d_2.weight = Parameter(w2)
       

        
        self.conv2d_1.register_full_backward_hook(print_grad)
        self.conv2d_2.register_full_backward_hook(print_grad)
        self.maxpool1.register_full_backward_hook(print_grad)
        self.maxpool2.register_full_backward_hook(print_grad)

    def forward(self, x):
        out = self.conv2d_1(x)
        #print("ref 1st out\n", out)
        out = self.maxpool1(out)
        #print("ref mxp1 out\n", out)
        out = self.conv2d_2(out)
        #print("ref 2nd out\n", out)
        out = self.maxpool2(out)
        #print("ref mxp2 out\n", out)
      
        return out




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=1, 
                                  out_channels=1, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  )   
        self.mxp1 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_2 = conv2d.TiledConv2d(in_channels=1, 
                                        out_channels=1, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )

        self.mxp2 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.block1 = sequential.mSequential(*[self.conv2d_1, self.mxp1, self.conv2d_2, self.mxp2])
        
    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).is_cuda
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, 1, 1)
        #print("!!!!!!!", model_device)
        stream_structure = self.block1

        # prepare grad info for correctness check(only for linear )
        i = len(li)
        for op in self.block1._modules.values():
            grad_dict_bk[id(op)*-1] = li[i-1]
            i -= 1

        

        

        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = [i,j]
                print("coord", coord)
                # TODO: here we have to somehow provide static info and num_conv. 
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict)
    # add grad_payload as negate keys
                info[0].update(grad_dict_bk)
      # add grad_payload as negate keys
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                input_tile = self.tsplit(x, info, stream_structure[0], model_device, [nTh-1, nTw-1]) # -1 here is to match 0-base
                print("***input tile", input_tile.size())
                out_temp = self.conv2d_1(input_tile, info)
                print("1 out_temp", out_temp[0].size())

                out_temp = self.mxp1(out_temp)
                print("max 1", out_temp[0].size())

                out_temp = self.conv2d_2(out_temp)
                print("2 out_temp", out_temp[0].size())
                out_temp = self.mxp2(out_temp)
                print("max 2", out_temp[0].size())

                
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
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    H = 144
    W = 144
    nTh = 3
    nTw = 3
    input = torch.rand(1,1,H,W, requires_grad = True)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    w1 = model.conv2d_1.weight.data
    w2 = model.conv2d_2.weight.data
    
    model_ref =  Net_ref(w1, w2).to(device)
    input_ref = input.data.clone() 
    input_ref = input_ref.cuda()
    input_ref.requires_grad = True
    out_ref = model_ref(input_ref)
    print("done ref")
    out_ref.sum().backward()




    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    out = model(input, H, W, nTh, nTw )


    

    # print("out shape", out)
    # print("out_ref ", out_ref)

    #print(input_ref.grad)
    # print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    # print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    # out.sum().backward()
    # # print("input ref grad", input_ref.grad)
    # # print("input grad", input.grad)
    # print("~~ check forward correctness ~~")
    # oH = out.size()[2]
    # oW = out.size()[3]
    # not_same_num = correctness_check.point_wise_compare_4d(1,1,oH, oW, out, out_ref)
    


    # print("#### compare grad_in")
    # #not_same_num = correctness_check.point_wise_compare_4d(1,1,H, W, input.grad, input_ref.grad.to('cpu'))
    # # print(type(li[0][0][0]))
    # not_same_num = correctness_check.point_wise_compare_4d(1,1,H, W, input.grad, li[-1][0][0].to('cpu'))
    

    # # print("w1 ref grad", model_ref.conv2d_1.weight.grad)
    # # print("w1 grad", model.conv2d_1.weight.grad)
    # print("#### compare w1")
    # not_same_num = correctness_check.point_wise_compare_4d(1,1,Kh,Kw, model_ref.conv2d_1.weight.grad, model.conv2d_1.weight.grad)

    # # print("w2 ref grad", model_ref.conv2d_2.weight.grad)
    # # print("w2 grad", model.conv2d_2.weight.grad)
    # print("#### compare w2")
    # not_same_num = correctness_check.point_wise_compare_4d(1,1,Kh,Kw, model_ref.conv2d_2.weight.grad, model.conv2d_2.weight.grad)

if __name__=="__main__":
    main()